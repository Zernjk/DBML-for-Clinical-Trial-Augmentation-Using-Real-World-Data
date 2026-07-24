import numpy as np
import pandas as pd
import os
import glob


def intersection_mask(
    q,
    loss_dict_trt, 
    loss_dict_big,
    n_same):

    record = {"Quantile": q}
    if q>1.0:
        qvals = {k: q*np.max(loss_dict_trt[k]) for k in ['binary', 'ordinal', 'cont','latent'] if k in loss_dict_trt}
    else:
        qvals = {k: np.quantile(loss_dict_trt[k], q) for k in ['binary', 'ordinal', 'cont','latent'] if k in loss_dict_trt}
    mask = np.ones(len(loss_dict_big['binary']), dtype=bool)

    for k in ['binary', 'ordinal', 'cont','latent']:
        if k in loss_dict_trt:
            mask &= (loss_dict_big[k] <= qvals[k])
            if n_same:
                record[f'Remains in {k} modality'] = f"{np.sum(loss_dict_big[k] <= qvals[k])} ({np.sum(loss_dict_big[k][:n_same] <= qvals[k])})"
            else:
                record[f'Remains in {k} modality'] = np.sum(loss_dict_big[k] <= qvals[k])
        else:
            continue
    return mask, record


def load_first_match(pattern):
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return matches[0]  # take the first match


def run_outlier_detection_intersection(
    x_trt_combined,
    loss_dict_trt, 
    loss_dict_big, 
    org_x_big,
    processed_x_big, 
    latent_big,
    z_mean_combined_big,
    output_dir, 
    quantiles=1.0,
    return_pc = True,
    n_same=None,
    train_vae = True,
    adjust=True,
):
    os.makedirs(output_dir, exist_ok=True)

    detection_records = []
    q = quantiles

    if train_vae:
        mask, record = intersection_mask(q, loss_dict_trt, loss_dict_big, n_same)

        if adjust:
            while np.sum(mask) < 20 * x_trt_combined.shape[0] and np.sum(mask) < int(1e4):
                q = q+0.05
                mask, record = intersection_mask(q, loss_dict_trt, loss_dict_big, n_same)

        record["Remains intersection"] = np.sum(mask)

        if n_same:
            record["Remains same distribution"] = np.sum(mask[:n_same])
        
        detection_records.append(record)

        np.savetxt(os.path.join(output_dir, f'org_x_pc_{q}.csv'), org_x_big[mask], delimiter=',')
        np.savetxt(os.path.join(output_dir, f'x_pc_{q}.csv'), processed_x_big[mask], delimiter=',')
        np.savetxt(os.path.join(output_dir, f'latent_pc_{q}.csv'), latent_big[mask], delimiter=',')
        np.savetxt(os.path.join(output_dir, f'ID_{q}.csv'), mask, delimiter=',')
        np.savetxt(os.path.join(output_dir, f'z_mean_combined_{q}.csv'), z_mean_combined_big[mask], delimiter=',')

        # Convert results to a DataFrame for a table
        detection_df = pd.DataFrame(detection_records)
        print(detection_df)

        # save the result table
        detection_df.to_csv(os.path.join(output_dir, 'outlier result.csv'), index=False)

        if return_pc:
            return mask, org_x_big[mask], processed_x_big[mask], z_mean_combined_big[mask], latent_big[mask]
        
    else:
        # Load files with "starts with" matching
        x_pc_id = np.loadtxt(load_first_match(os.path.join(output_dir, 'ID_*.csv')), delimiter=',')
        org_x_pc = np.loadtxt(load_first_match(os.path.join(output_dir, 'org_x_pc_*.csv')), delimiter=',')
        processed_x_pc = np.loadtxt(load_first_match(os.path.join(output_dir, 'x_pc_*.csv')), delimiter=',')
        z_mean_combined_pc = np.loadtxt(load_first_match(os.path.join(output_dir, 'z_mean_combined_*.csv')), delimiter=',')
        latent_pc = np.loadtxt(load_first_match(os.path.join(output_dir, 'latent_pc_*.csv')), delimiter=',')
        
        detection_df = pd.read_csv(os.path.join(output_dir, 'outlier result.csv'))
        print(detection_df)

        if return_pc:
            return x_pc_id, org_x_pc, processed_x_pc, z_mean_combined_pc, latent_pc