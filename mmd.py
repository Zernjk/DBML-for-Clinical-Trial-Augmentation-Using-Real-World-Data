import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from vae_trainers import get_cosine_schedule_with_warmup
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from itertools import cycle


def rbf_kernel_matrix_torch(x, y, sigma=1.0):
    """
    Compute the RBF kernel matrix between x and y using PyTorch.
    """
    dist_sq = torch.cdist(x, y, p=2).pow(2)
    K = torch.exp(-dist_sq / (2 * sigma ** 2))
    return K


def get_rbf_loss(
    Kxy,
    Kyy,
    logits,
    M,
    N,
    device,
    ssl_penalty=False,
    lambda0=5,
    lambda1=0.1,
    p_star=None,
    adjust=False
):
    w_raw = torch.nn.functional.softplus(logits) if adjust else torch.sigmoid(logits)
    w = w_raw / (torch.sum(w_raw) + 1e-10)

    r1 = torch.sum(torch.matmul(Kxy, w.view(N, 1)))

    # Include diagonal elements
    r2 = torch.sum(
        torch.matmul(w.view(1, N), torch.matmul(Kyy, w.view(N, 1)))
    )

    obj = r2 - 2*r1 / M 

    if adjust:
        sample_weights = w * M
        obj += 100.0 * torch.sum(torch.relu(sample_weights - 1.0) ** 2)

    if ssl_penalty:
        penalty_params = logits if adjust else w_raw
        if p_star is not None:
            obj += 1e-6 * penalty_loss(
                lambda1=lambda1,
                lambda0=lambda0,
                p_star=p_star,
                params=penalty_params
            )
        else:
            l1_coeff = 1e-6 * 0.5 * (lambda0 + lambda1)
            obj += l1_coeff * penalty_params.abs().sum()

    return obj


def get_euclidean_loss(
    Dxy,
    Dxx,
    logits,
    M,
    N,
    device,
    ssl_penalty=False,
    lambda0=5,
    lambda1=0.1,
    p_star=None,
    adjust=False
):
    w_raw = torch.nn.functional.softplus(logits) if adjust else torch.sigmoid(logits)
    w_norm = w_raw / (torch.sum(w_raw) + 1e-10)

    r1 = torch.sum(w_norm.view(-1, 1) * Dxy) / M

    outer_w = torch.outer(w_norm, w_norm)
    r2 = torch.sum(outer_w * Dxx)

    obj = 2 * r1 - r2

    if adjust:
        sample_weights = w_norm * M
        obj += 100.0 * torch.sum(torch.relu(sample_weights - 1.0) ** 2)

    if ssl_penalty:
        penalty_params = logits if adjust else w_raw
        if p_star is not None:
            obj += 1e-6 * penalty_loss(
                lambda1=lambda1,
                lambda0=lambda0,
                p_star=p_star,
                params=penalty_params
            )
        else:
            l1_coeff = 1e-6 * 0.5 * (lambda0 + lambda1)
            obj += l1_coeff * penalty_params.abs().sum()

    return obj


def penalty_loss(lambda1=0.1, lambda0=5, p_star=None, params=None):
    loss = (lambda1 * p_star + lambda0 * (1 - p_star)) * params.abs()
    return loss.sum()


def run_mmd(x_trt, 
            x_pc, 
            device, 
            train,
            save_results,
            save_detail,
            output_dir,
            lr,
            sigma,
            ssl_penalty,
            lambda0,
            lambda1,
            num_iterations,
            scheduler,
            kernel='rbf',
            adjust=False,
            return_result=True):
    
    if kernel == 'eucl':
        kernel = 'euclidean'
    if kernel not in ('rbf', 'euclidean'):
        raise ValueError("kernel must be 'rbf', 'euclidean', or 'eucl'")

    M, N = x_trt.shape[0], x_pc.shape[0]
    kernel_name = "RBF Kernel (KMMD)" if kernel == 'rbf' else "Euclidean Kernel (Energy Balancing)"
    adjustment_name = " with penalty-to-one adjustment" if adjust else ""
    
    if ssl_penalty:
        print(
            f"Do {kernel_name}{adjustment_name} on {M} trts with {N} possible crls "
            f"with SSL penalty lambda0={lambda0}, lambda1={lambda1}..."
        )
    else:
        print(f"Do {kernel_name}{adjustment_name} on {M} trts with {N} possible crls...")

    os.makedirs(output_dir, exist_ok=True)

    if train:
        x_all = np.vstack([x_pc, x_trt])
        stds = np.std(x_all, axis=0, ddof=1)
        stds[stds == 0] = 1.0
        
        x_trt_scaled = (x_trt - np.mean(x_all, axis=0)) / stds
        x_pc_scaled = (x_pc - np.mean(x_all, axis=0)) / stds

        x = torch.tensor(x_trt_scaled, dtype=torch.float64, device=device)
        y = torch.tensor(x_pc_scaled, dtype=torch.float64, device=device)

        if kernel == 'rbf':
            Mxy = rbf_kernel_matrix_torch(x, y, sigma)
            Mxx = rbf_kernel_matrix_torch(y, y, sigma) # Mxx is actually Kyy (pc-pc)
        elif kernel == 'euclidean':
            Mxy = torch.cdist(y, x, p=2) # Dxy: (N, M)
            Mxx = torch.cdist(y, y, p=2) # Dxx: (N, N)
        else:
            raise ValueError("kernel must be 'rbf' or 'euclidean'")

        # Initialize parameters to optimize
        params = torch.randn(N, dtype=torch.float64, device=device, requires_grad=True) if kernel == 'rbf' else \
                 torch.zeros(N, dtype=torch.float64, device=device, requires_grad=True)
                 
        p_star = None
        if ssl_penalty:
            p_star = 0.5 * torch.ones(N, dtype=torch.float64, device=device, requires_grad=False)

        optimizer = torch.optim.Adam([params], lr=lr)
        num_warmup_steps = int(0.1 * num_iterations)

        if scheduler == 'linear':
            scheduler_fn = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_iterations
            )
        else:
            scheduler_fn = get_cosine_schedule_with_warmup(
                optimizer, warmup_epochs=num_warmup_steps, total_epochs=num_iterations
            )

        iterations = []
        loss_values = []

        for i in range(num_iterations):
            optimizer.zero_grad()

            if kernel == 'rbf':
                loss = get_rbf_loss(
                    Mxy,
                    Mxx,
                    params,
                    M,
                    N,
                    device,
                    ssl_penalty,
                    lambda0,
                    lambda1,
                    p_star,
                    adjust
                )
            else:
                loss = get_euclidean_loss(
                    Mxy,
                    Mxx,
                    params,
                    M,
                    N,
                    device,
                    ssl_penalty,
                    lambda0,
                    lambda1,
                    p_star,
                    adjust
                )

            loss.backward()
            optimizer.step()
            scheduler_fn.step()

            if i % 2000 == 0:
                iterations.append(i)
                loss_values.append(loss.item())
                print(f"Iteration {i}: Loss = {loss.item()}")

        print(f"\nFinal loss after {num_iterations} iterations:")
        print(f"Loss = {loss.item()}")

        beta_opt = params.detach()
        if adjust:
            w_opt_raw = torch.nn.functional.softplus(beta_opt).cpu().numpy()
            w_opt = w_opt_raw / (np.sum(w_opt_raw) + 1e-10) * M
        else:
            w_opt_raw = torch.sigmoid(beta_opt).cpu().numpy()
            w_opt = w_opt_raw / (np.mean(w_opt_raw) + 1e-10)

        if save_results:
            if kernel == 'rbf' and adjust:
                param_file = 'rbf_params_ssl.pt' if ssl_penalty else 'rbf_params_pento1.pt'
                w_file = 'w_rbf_ssl.csv' if ssl_penalty else 'w_rbf_pento1.csv'
            elif kernel == 'rbf':
                param_file = 'kmmd_params_ssl.pt' if ssl_penalty else 'kmmd_params.pt'
                w_file = 'w_opt_ssl.csv' if ssl_penalty else 'w_opt.csv'
            elif adjust:
                param_file = 'energybal_params_pento1.pt'
                w_file = 'w_engy_pento1.csv'
            else:
                param_file = 'energybal_params.pt'
                w_file = 'w_engy.csv'
                
            torch.save(params.detach().cpu(), os.path.join(output_dir, param_file))
            np.savetxt(os.path.join(output_dir, w_file), w_opt, delimiter=',', fmt='%f')

        if save_detail:
            plt.plot(iterations, loss_values)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title(f"{kernel_name} Loss over Iterations")
            plt.grid(True)
            if kernel == 'rbf':
                loss_img = 'rbf_kernel_loss_pento1.png' if adjust else 'kmmd_loss.png'
            else:
                loss_img = 'energybal_loss_pento1.png' if adjust else 'energybal_loss.png'
            plt.savefig(os.path.join(output_dir, loss_img), dpi=300)
            plt.close() 

    else:
        if kernel == 'rbf' and adjust:
            w_file = 'w_rbf_ssl.csv' if ssl_penalty else 'w_rbf_pento1.csv'
        elif kernel == 'rbf':
            w_file = 'w_opt_ssl.csv' if ssl_penalty else 'w_opt.csv'
        elif adjust:
            w_file = 'w_engy_pento1.csv'
        else:
            w_file = 'w_engy.csv'
        w_opt = np.loadtxt(os.path.join(output_dir, w_file), delimiter=',')

    if return_result:
        return w_opt


class WeightNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, x):
        return self.network(x)


def run_mmd_nn(x_trt, 
               x_pc, 
               device, 
               train,
               save_results,
               save_detail,
               output_dir,
               lr,
               sigma,
               ssl_penalty,
               lambda0,
               lambda1,
               num_iterations,
               scheduler,
               batch_size_trt=128,
               batch_size_pc=128,
               hidden_dim=64,
               kernel='rbf',
               adjust=False,
               return_result=True):
    
    if kernel == 'eucl':
        kernel = 'euclidean'
    if kernel not in ('rbf', 'euclidean'):
        raise ValueError("kernel must be 'rbf', 'euclidean', or 'eucl'")

    M, D_trt = x_trt.shape
    N, D_pc = x_pc.shape
    
    if D_trt != D_pc:
        raise ValueError(f"Feature dimensions must match: trt={D_trt}, pc={D_pc}")
    
    D = D_trt 
    kernel_name = "RBF Kernel (KMMD-NN)" if kernel == 'rbf' else "Euclidean Kernel (EnergyBal-NN)"
    adjustment_name = " with penalty-to-one adjustment" if adjust else ""

    if ssl_penalty:
        print(
            f"Do {kernel_name}{adjustment_name} on {M} trts with {N} possible crls "
            f"(SSL penalty lambda0={lambda0}, lambda1={lambda1})..."
        )
    else:
        print(f"Do {kernel_name}{adjustment_name} on {M} trts with {N} possible crls...")

    os.makedirs(output_dir, exist_ok=True)

    if kernel == 'rbf' and (batch_size_trt < 2 or batch_size_pc < 2):
        raise ValueError(
            f"Batch sizes must be >= 2 for KMMD calculation. "
            f"Got trt={batch_size_trt}, pc={batch_size_pc}. "
        )

    weight_net = WeightNetwork(input_dim=D, hidden_dim=hidden_dim).to(device).double()
    if kernel == 'rbf':
        nn_model_file = 'kmmd_nn_pento1_model.pt' if adjust else 'kmmd_nn_model.pt'
    else:
        nn_model_file = 'energybal_nn_pento1_model.pt' if adjust else 'energybal_nn_model.pt'
    nn_model_path = os.path.join(output_dir, nn_model_file)

    if train:
        x_all = np.vstack([x_pc, x_trt])
        stds = np.std(x_all, axis=0, ddof=1)
        stds[stds == 0] = 1.0
        
        x_trt_scaled = (x_trt - np.mean(x_all, axis=0)) / stds
        x_pc_scaled = (x_pc - np.mean(x_all, axis=0)) / stds

        x_trt_tensor = torch.tensor(x_trt_scaled, dtype=torch.float64)
        x_pc_tensor = torch.tensor(x_pc_scaled, dtype=torch.float64)

        trt_dataset = TensorDataset(x_trt_tensor)
        pc_dataset = TensorDataset(x_pc_tensor)

        trt_loader = DataLoader(trt_dataset, batch_size=batch_size_trt, shuffle=True, drop_last=False)
        pc_loader = DataLoader(pc_dataset, batch_size=batch_size_pc, shuffle=True, drop_last=False)

        trt_iter = cycle(trt_loader)
        pc_iter = cycle(pc_loader)

        optimizer = torch.optim.Adam(weight_net.parameters(), lr=lr)
        num_warmup_steps = int(0.1 * num_iterations)

        if scheduler == 'linear':
            scheduler_fn = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_iterations
            )
        else:
            scheduler_fn = get_cosine_schedule_with_warmup(
                optimizer, warmup_epochs=num_warmup_steps, total_epochs=num_iterations
            )

        iterations = []
        loss_values = []
        weight_net.train()

        for i in range(num_iterations):
            optimizer.zero_grad()

            x_trt_batch = next(trt_iter)[0].to(device)
            x_pc_batch = next(pc_iter)[0].to(device)

            B_M, B_N = x_trt_batch.shape[0], x_pc_batch.shape[0]
            
            if B_N <= 1:
                continue 

            if kernel == 'rbf':
                Mxy_batch = rbf_kernel_matrix_torch(x_trt_batch, x_pc_batch, sigma)
                Mxx_batch = rbf_kernel_matrix_torch(x_pc_batch, x_pc_batch, sigma)
            elif kernel == 'euclidean':
                Mxy_batch = torch.cdist(x_pc_batch, x_trt_batch, p=2)
                Mxx_batch = torch.cdist(x_pc_batch, x_pc_batch, p=2)
            else:
                raise ValueError("kernel must be 'rbf' or 'euclidean'")

            logits_pc_batch = weight_net(x_pc_batch).squeeze()

            if kernel == 'rbf':
                loss = get_rbf_loss(
                    Mxy_batch,
                    Mxx_batch,
                    logits_pc_batch,
                    B_M,
                    B_N,
                    device,
                    ssl_penalty,
                    lambda0,
                    lambda1,
                    adjust=adjust
                )
            else:
                loss = get_euclidean_loss(
                    Mxy_batch,
                    Mxx_batch,
                    logits_pc_batch,
                    B_M,
                    B_N,
                    device,
                    ssl_penalty,
                    lambda0,
                    lambda1,
                    adjust=adjust
                )

            loss.backward()
            optimizer.step()
            scheduler_fn.step()

            if i % 2000 == 0:
                iterations.append(i)
                loss_values.append(loss.item())
                print(f"Iteration {i}: Loss = {loss.item()}")

        print(f"\nFinal loss after {num_iterations} iterations:")
        print(f"Loss = {loss.item()}")

        if save_results:
            torch.save(weight_net.state_dict(), nn_model_path)
            print(f"Saved network state to {nn_model_path}")

        if save_detail:
            plt.plot(iterations, loss_values)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title(f"{kernel_name} Loss over Iterations")
            plt.grid(True)
            if kernel == 'rbf':
                loss_img = 'kmmd_nn_loss_pento1.png' if adjust else 'kmmd_nn_loss.png'
            else:
                loss_img = 'energybal_nn_loss_pento1.png' if adjust else 'energybal_nn_loss.png'
            plt.savefig(os.path.join(output_dir, loss_img), dpi=300)
            plt.close()

    else:
        try:
            weight_net.load_state_dict(torch.load(nn_model_path, map_location=device))
            print(f"Loaded trained network from {nn_model_path}")
        except FileNotFoundError:
            print(f"Error: No trained model found at {nn_model_path}. Please run with train=True first.")
            return None

    weight_net.eval()
    
    x_all = np.vstack([x_pc, x_trt])
    stds = np.std(x_all, axis=0, ddof=1)
    stds[stds == 0] = 1.0
    x_pc_scaled = (x_pc - np.mean(x_all, axis=0)) / stds

    w_opt_raw = np.zeros(N)
    inference_loader = DataLoader(TensorDataset(torch.tensor(x_pc_scaled, dtype=torch.float64)), 
                                  batch_size=batch_size_pc * 4)
    
    start_idx = 0
    with torch.no_grad():
        for x_pc_batch in inference_loader:
            x_batch_dev = x_pc_batch[0].to(device)
            logits_batch = weight_net(x_batch_dev)
            if adjust:
                w_batch = torch.nn.functional.softplus(logits_batch).squeeze().cpu().numpy()
            else:
                w_batch = torch.sigmoid(logits_batch).squeeze().cpu().numpy()
            
            if w_batch.ndim == 0:
                w_batch = np.array([w_batch])
                
            end_idx = start_idx + len(w_batch)
            w_opt_raw[start_idx:end_idx] = w_batch
            start_idx = end_idx

    if adjust:
        w_opt = w_opt_raw / (np.sum(w_opt_raw) + 1e-10) * M
    else:
        w_opt = w_opt_raw / (np.mean(w_opt_raw) + 1e-10)

    if save_results and train:
        w_file = 'w_opt_nn_pento1.csv' if adjust else 'w_opt_nn.csv'
        w_opt_path = os.path.join(output_dir, w_file)
        np.savetxt(w_opt_path, w_opt, delimiter=',', fmt='%f')
        print(f"Saved final {N} weights to {w_opt_path}")

    if return_result:
        return w_opt
