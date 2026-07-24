import numpy as np
import os

def combine_latent_embeddings(z_mean_list, z_log_var_list, output_dir):
    z_mean_combined = np.concatenate(z_mean_list, axis=1)
    z_log_var_combined = np.concatenate(z_log_var_list, axis=1)
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, 'z_mean_combined.csv'), z_mean_combined, delimiter=',')
    # np.savetxt(os.path.join(output_dir, 'z_log_var_combined.csv'), z_log_var_combined, delimiter=',')
    return z_mean_combined, z_log_var_combined
