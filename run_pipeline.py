from vae_trainers import train_vae_on_modality
from latent_utils import combine_latent_embeddings
from outlier_detection import run_outlier_detection_intersection
from data_preprocessing import preprocess_data
from mmd import run_mmd, run_mmd_nn
import numpy as np
import time

def run_full_simulation_from_data(
    data,
    big_data=None,
    configs={},  # dictionary with training params per modality
    output_root='./output',
    train_vae=True,
    adjust = False,
    save_vae_model=True,
    save_vae_results=True,
    save_vae_detail=True,
    train_kmmd=True,
    save_kmmd_results=True,
    save_kmmd_detail = True,
    return_result_type='processed',
    device='cuda'
):
    
    print("Data Preprocessing...")
    time1 = time.time()

    preprocess_config = configs.get('preprocess', {})
    binary_indices = preprocess_config.get('binary_indices', [])
    categorical_indices_with_levels = preprocess_config.get('categorical_indices_with_levels', {})
    ordinal_indices = preprocess_config.get('ordinal_indices',[])
    continuous_indices = preprocess_config.get('continuous_indices', [])

    processed, metadata = preprocess_data(data, binary_indices, ordinal_indices, continuous_indices, categorical_indices_with_levels)
    processed_big, _ = preprocess_data(big_data, binary_indices, ordinal_indices, continuous_indices, categorical_indices_with_levels)
    x_trt_combined = np.concatenate([processed[k] for k in ['binary', 'ordinal', 'cont']], axis=1)
    x_big_combined = np.concatenate([processed_big[k] for k in ['binary', 'ordinal', 'cont']], axis=1)

    time2 = time.time()

    results = {}
    print("Traning VAE stage 1...")
    
    time3 = time.time()

    for name, loss_type in zip(['binary', 'ordinal', 'cont'], ['binary', 'ordinal', 'mse']):
        if processed[name].shape[1] == 0:
            print(f"[INFO] No '{name}' modality found. Skipping...")
            results[name] = {
                'z_mean': np.empty((len(processed[name]), 0)),
                'z_log_var': np.empty((len(processed[name]), 0)),
                'loss_trt': np.zeros(len(processed[name])),
                'z_mean_big': np.empty((len(processed_big[name]), 0)),
                'z_log_var_big': np.empty((len(processed_big[name]), 0)),
                'loss_big': np.zeros(len(processed_big[name]))
            }
            continue

        print(f"Training VAE for {name} modality...")

        out_dir = f"{output_root}/vae/{name}"

        vae_config = configs.get('vae', {}).get(name, {})
        result = train_vae_on_modality(
            x=processed[name],
            loss_type=loss_type,
            output_dir=out_dir,
            device=device,
            batch_size=vae_config.get('batch_size', 128),
            epochs=vae_config.get('epochs', 600),
            hidden_dim=vae_config.get('hidden_dim', 50),
            n_components=vae_config.get('n_components', 3),
            lr=vae_config.get('lr', 0.05),
            scheduler=vae_config.get('scheduler','linear'),
            ordinal_K=vae_config.get('ordinal_K', 4),
            embedding_dim=vae_config.get('embedding_dim', 4),
            genuine_binary_indices=(
                metadata['genuine_binary_indices'] if name == 'binary' else None
            ),
            categorical_groups=(
                metadata['categorical_groups'] if name == 'binary' else None
            ),
            big_data=processed_big[name],
            train=train_vae,
            save_model=save_vae_model,
            save_results=save_vae_results,
            save_detail=save_vae_detail
        )
        results[name] = result

        print(f"Finishing VAE for {name} modality...")

    z_mean_combined, _ = combine_latent_embeddings(
        [results[k]['z_mean'] for k in ['binary', 'ordinal', 'cont'] if results[k]['z_mean'].shape[1] > 0],
        [results[k]['z_log_var'] for k in ['binary', 'ordinal', 'cont'] if results[k]['z_log_var'].shape[1] > 0],
        f"{output_root}/vae/latent_vae/trt"
    )

    z_mean_combined_big, _ = combine_latent_embeddings(
        [results[k]['z_mean_big'] for k in ['binary', 'ordinal', 'cont'] if results[k]['z_mean_big'].shape[1] > 0],
        [results[k]['z_log_var_big'] for k in ['binary', 'ordinal', 'cont'] if results[k]['z_log_var_big'].shape[1] > 0],
        f"{output_root}/vae/latent_vae/big"
    )

    print("Traning VAE stage 2...")
    print("Training latent VAE on concatenated embeddings...")
    
    z_input = z_mean_combined
    latent_out_dir = f"{output_root}/vae/latent_vae"
    latent_config =  configs.get('vae', {}).get('latent', {})
    latent_result = train_vae_on_modality(
        x=z_input,
        loss_type='mse',
        output_dir=latent_out_dir,
        device=device,
        batch_size=latent_config.get('batch_size', 128),
        epochs=latent_config.get('epochs', 600),
        hidden_dim=latent_config.get('hidden_dim', 50),
        n_components=latent_config.get('n_components', 3),
        lr=latent_config.get('lr', 0.05),
        scheduler=latent_config.get('scheduler','linear'),
        big_data=z_mean_combined_big,
        train=train_vae,
        save_model=save_vae_model,
        save_results=save_vae_results,
        save_detail=save_vae_detail
    )
    results["latent"] = latent_result
    
    print("Finishing VAE stage 2...")

    print("Running outlier detection based on trained models and big data...")

    x_pc_id, org_x_pc, processed_x_pc, z_mean_combined_pc, latent_pc = run_outlier_detection_intersection(
        x_trt_combined = x_trt_combined,
        loss_dict_trt={k: results[k]['loss_trt'] for k in results if results[k]['z_mean'].shape[1] > 0},
        loss_dict_big={k: results[k]['loss_big'] for k in results if results[k]['z_mean'].shape[1] > 0},
        org_x_big = big_data,
        processed_x_big=x_big_combined,
        latent_big = results['latent']['z_mean_big'],
        z_mean_combined_big = z_mean_combined_big,
        output_dir=f"{output_root}/vae/intersection",
        n_same=configs.get('detection',{}).get('n_same', 10000),
        train_vae=train_vae,
        adjust = adjust
    )
    
    time4 = time.time()

    kmmd_config = configs.get('kmmd', {})
    kmmd_method = kmmd_config.get('method', 'rbf')
    kmmd_latent = kmmd_config.get('use_latent', False)
    kmmd_adjust = kmmd_config.get('adjust', False)

    adjustment_name = " with penalty-to-one adjustment" if kmmd_adjust else ""
    print(f"Running {kmmd_method}{adjustment_name} on selected possible controls...")

    # Arguments shared by run_mmd and run_mmd_nn in mmd.py.
    common_kmmd_args = {
        'x_trt': results["latent"]["z_mean"] if kmmd_latent else x_trt_combined,
        'x_pc': latent_pc if kmmd_latent else processed_x_pc,
        'device': device,
        'train': train_kmmd,
        'save_results': save_kmmd_results,
        'save_detail': save_kmmd_detail,
        'output_dir': f"{output_root}/kmmd/latent" if kmmd_latent else f"{output_root}/kmmd",
        'lr': kmmd_config.get('lr', 0.01),
        'sigma': kmmd_config.get('sigma', 1.0),
        'ssl_penalty': kmmd_config.get('ssl_penalty', False),
        'lambda0': kmmd_config.get('lambda0', 5),
        'lambda1': kmmd_config.get('lambda1', 0.1),
        'num_iterations': kmmd_config.get('epochs', int(5e4)),
        'scheduler': kmmd_config.get('scheduler', 'linear'),
        'adjust': kmmd_adjust
        # return_result is True by default in both, so no need to pass
    }

    w_opt = None
    if kmmd_method == 'nn':
        print("Using RBF kernel with neural-network weights (mini-batch method).")
        
        # Get NN-specific parameters from config
        w_opt = run_mmd_nn(
            **common_kmmd_args,
            batch_size_trt=kmmd_config.get('batch_size_trt', 128),
            batch_size_pc=kmmd_config.get('batch_size_pc', 128),
            hidden_dim=kmmd_config.get('hidden_dim', 64),
            kernel='rbf'
        )

    elif kmmd_method == 'rbf':
        print("Using RBF kernel (full-batch method).")
        
        w_opt = run_mmd(
            **common_kmmd_args,
            kernel='rbf'
        )

    elif kmmd_method in ('euclidean', 'eucl'):
        print("Using Euclidean distance (full-batch method).")
        w_opt = run_mmd(
            **common_kmmd_args,
            kernel='euclidean'
        )

    else:
        raise ValueError(f"Unknown kmmd method in config: {kmmd_method}. "
                         "Must be 'rbf', 'nn', 'euclidean', or 'eucl'.")

    # Now w_opt holds the weights from whichever method was run
    if w_opt is not None:
        print(f"Successfully computed {len(w_opt)} KMMD weights.")
        
    time5 = time.time()
    
    # print time consuming
    print(f"Data Processing time: {time2-time1:.2f} seconds")
    print(f"Anomaly Detection time: {time4-time3:.2f} seconds ({(time4-time3)/60:.2f} minutes)")
    print(f"KMMD time: {time5-time4:.2f} seconds ({(time5-time4)/60:.2f} minutes)")

    
    if return_result_type=='original':
        return w_opt, x_pc_id, x_trt_combined, x_big_combined, processed_x_pc, data, big_data, org_x_pc

    if return_result_type=='processed':
        return w_opt, x_pc_id, x_trt_combined, x_big_combined, processed_x_pc, x_trt_combined, x_big_combined, processed_x_pc
    
    if return_result_type=='z_combined':
        return w_opt, x_pc_id, x_trt_combined, x_big_combined, processed_x_pc, z_mean_combined, z_mean_combined_big, z_mean_combined_pc
    
    if return_result_type=='latent':
        return w_opt, x_pc_id, x_trt_combined, x_big_combined, processed_x_pc, results['latent']['z_mean'], results['latent']['z_mean_big'], latent_pc
