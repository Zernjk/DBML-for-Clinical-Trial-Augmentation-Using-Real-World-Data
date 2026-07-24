import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.optim.lr_scheduler as lr_scheduler

from models_ebd import *

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        return 0.5 * (1. + np.cos(np.pi * (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    return LambdaLR(optimizer, lr_lambda)

def evaluate_vae_on_data(model, x, loss_type, device='cuda', save_detail=True):
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.long if loss_type == 'ordinal' else torch.float32).to(device)

    with torch.no_grad():
        z_mean, z_log_var = model.encode(x_tensor)
        x_recon = model.decode(z_mean)

        if loss_type == 'ordinal':
            thetas = compute_thetas(model.rhos.detach().cpu().numpy())
            log_sigmas = 0
        elif loss_type == 'mse':
            log_sigmas = model.log_sigmas
            thetas = 0
        else:
            log_sigmas = thetas = 0

        loss = model.reconstruction_loss_outlier(x_recon, x_tensor, log_sigmas, thetas).detach().cpu().numpy()

    return z_mean.detach().cpu().numpy(), z_log_var.detach().cpu().numpy(), loss

def plot_recon(x, x_recon, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Convert tensors to numpy
    x_org = x.detach().cpu().numpy()
    x_recon = x_recon.detach().cpu().numpy()

    # Reshape to (n, d) even if only 1D
    x_org = np.asarray(x_org).reshape(len(x_org), -1)
    x_recon = np.asarray(x_recon).reshape(len(x_recon), -1)

    input_dim = x_org.shape[1]
    n_cols = min(input_dim, 3)
    n_rows = math.ceil(input_dim / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(range(input_dim)):
        axes[i].scatter(x_org[:, col], x_recon[:, col], alpha=0.7)
        axes[i].set_title(f"Scatter Plot of Feature {col}")
        axes[i].set_xlabel("Original")
        axes[i].set_ylabel("Reconstructed")
        axes[i].grid(True)

    # Hide unused plots
    for j in range(input_dim, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Reconstruction.png'), dpi=300)
    plt.close()



def evaluate_vae_detail(model, x, x_recon, loss_type, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    plot_recon(x, x_recon, out_dir)

    if loss_type == 'binary':
        binary_x_recon = (x_recon.detach().cpu().numpy() >= 0.5).astype(int)
        accuracy = np.mean(x.cpu().numpy() == binary_x_recon, axis=0)
        np.savetxt(os.path.join(out_dir, 'accuracy.csv'), accuracy, delimiter=',')
        print(accuracy)

    elif loss_type == 'ordinal':
        # save rhos and thetas
        rhos_mat = rhos_to_matrix_numpy(model.rhos)
        np.savetxt(os.path.join(out_dir, 'rhos.csv'), rhos_mat, delimiter=',')
        thetas = compute_thetas(rhos_mat)
        np.savetxt(os.path.join(out_dir, 'thetas.csv'), thetas, delimiter=',')
        print('thetas:\n', thetas)
        Q_mat = compute_ordinal_quantile(x_recon.detach().cpu().numpy(), thetas)
        np.savetxt(os.path.join(out_dir, 'Q.csv'), Q_mat, delimiter=',')
        print('Quantile:\n', Q_mat)

        # calculate the true quantile matrix
        Q_true = compute_ordinal_quantile_true(X=x.cpu().numpy(), num_classes=model.ordinal_K)
        np.savetxt(os.path.join(out_dir, 'Q_true.csv'), Q_true, delimiter=',')
        print('Quantile True:\n', Q_true)

        ordinal_x_recon = compute_ordinal(x_recon.detach().cpu().numpy(), thetas)
        accuracy = np.mean(x.cpu().numpy() == ordinal_x_recon, axis=0)
        np.savetxt(os.path.join(out_dir, 'accuracy.csv'), accuracy, delimiter=',')
        print('accuracy:', accuracy) 

    else:
        mse = np.mean((x.cpu().numpy() - x_recon.detach().cpu().numpy())**2, axis=0)
        np.savetxt(os.path.join(out_dir, 'mse.csv'), mse, delimiter=',')
        print(mse)    


def train_vae_on_modality(x, 
                          loss_type, 
                          output_dir, 
                          device='cuda',
                          batch_size=128, 
                          epochs=600, 
                          hidden_dim=50,
                          n_components=3, 
                          lr=0.005, 
                          scheduler = 'linear',
                          ordinal_K=4, 
                          embedding_dim=4,
                          proximal=True, 
                          train=True,
                          save_model=True, 
                          save_results=True,
                          save_detail = True,
                          big_data=None):
    
    x_tensor = torch.tensor(x, dtype=torch.long if loss_type == 'ordinal' else torch.float32)
    input_dim = x.shape[1]
    dataset = DataLoader(x_tensor, batch_size=batch_size, shuffle=False)

    sigma_prior_scale, sigmas_init = (get_sigma_prior(x_tensor) if loss_type == 'mse' else (None, None))

    model = VAE(
        batch_size=batch_size, input_dim=input_dim, latent_dim=n_components,
        hidden_dim=hidden_dim, loss_type=loss_type,
        ordinal_K=ordinal_K if loss_type == 'ordinal' else None,
        embedding_dim=embedding_dim if loss_type == 'ordinal' else None,
        sigma_prior_scale=sigma_prior_scale, sigmas_init=sigmas_init
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if scheduler == 'linear':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs=50, total_epochs=epochs)

    
    if train:
        for epoch in range(epochs):
            if epoch % 200 == 0:
                print("Training epoch: ", epoch)
            model.train()
            for batch in dataset:
                optimizer.zero_grad()
                features = batch.to(device)
                like, kl, sigma = model.vae_loss(features)
                total = like + kl + sigma if proximal else like + kl
                total.backward()
                optimizer.step()
            scheduler.step()

        os.makedirs(output_dir, exist_ok=True)
        if save_model:
            torch.save({'state_dict': model.state_dict()}, os.path.join(output_dir, 'model.pth'))
    else:
        model.load_state_dict(torch.load(os.path.join(output_dir, 'model.pth'))['state_dict'])

    model.eval()
    x_tensor = x_tensor.to(device)
    z_mean, z_log_var = model.encode(x_tensor)
    x_recon = model.decode(z_mean)
    rhos_matrix = rhos_to_matrix_numpy(model.rhos) if loss_type == 'ordinal' else 0
    thetas = compute_thetas(rhos_matrix) if loss_type == 'ordinal' else 0
    log_sigmas = model.log_sigmas if loss_type == 'mse' else 0
    loss_train = model.reconstruction_loss_outlier(x_recon, x_tensor, log_sigmas, thetas).detach().cpu().numpy()

    if save_results:
        np.savetxt(os.path.join(output_dir, 'z_mean.csv'), z_mean.detach().cpu().numpy(), delimiter=',')
        np.savetxt(os.path.join(output_dir, 'z_log_var.csv'), z_log_var.detach().cpu().numpy(), delimiter=',')
        np.savetxt(os.path.join(output_dir, 'loss_train.csv'), loss_train, delimiter=',')

    results = {
        'z_mean': z_mean.detach().cpu().numpy(),
        'z_log_var': z_log_var.detach().cpu().numpy(),
        'loss_trt': loss_train,
        'model': model
    }

    if save_detail:
        print("Save details on Trt...")
        evaluate_vae_detail(model=model, x=x_tensor, x_recon=x_recon, loss_type=loss_type, out_dir=f"{output_dir}/detail/trt")


    # If big_data is provided, evaluate and save
    if big_data is not None:

        model.eval()
        x_tensor = torch.tensor(big_data, dtype=torch.long if loss_type == 'ordinal' else torch.float32).to(device)
        z_mean_big, z_log_var_big = model.encode(x_tensor)
        x_recon = model.decode(z_mean_big)
        rhos_matrix = rhos_to_matrix_numpy(model.rhos) if loss_type == 'ordinal' else 0
        thetas = compute_thetas(rhos_matrix) if loss_type == 'ordinal' else 0
        log_sigmas = model.log_sigmas if loss_type == 'mse' else 0
        loss_big = model.reconstruction_loss_outlier(x_recon, x_tensor, log_sigmas, thetas).detach().cpu().numpy()

        #z_mean_big, z_log_var_big, loss_big = evaluate_vae_on_data(model, big_data, loss_type, device, save_detail)
        results['z_mean_big'] = z_mean_big.detach().cpu().numpy()
        results['z_log_var_big'] = z_log_var_big.detach().cpu().numpy()
        results['loss_big'] = loss_big

        if save_detail:
            print("Save details on Big...")
            evaluate_vae_detail(model=model, x=x_tensor, x_recon=x_recon, loss_type=loss_type, out_dir=f"{output_dir}/detail/big")
            log_loss_all = np.concatenate([results['loss_trt'], results['loss_big']],axis=0)
            log_loss_all = np.log(np.clip(log_loss_all, a_min=1e-20, a_max=None))
            plot_histograms(log_loss_all, [results['loss_trt'].shape[0]], column='reconstruction_loss', labels=['trt', 'big'], bins=20, output_path=f"{output_dir}/detail")


        if save_results:
            np.savetxt(os.path.join(output_dir, 'z_mean_big.csv'), z_mean_big.detach().cpu().numpy(), delimiter=',')
            np.savetxt(os.path.join(output_dir, 'z_log_var_big.csv'), z_log_var_big.detach().cpu().numpy(), delimiter=',')
            np.savetxt(os.path.join(output_dir, 'loss_big.csv'), loss_big, delimiter=',')

    return results