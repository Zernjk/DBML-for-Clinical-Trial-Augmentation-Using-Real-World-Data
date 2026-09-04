import torch
import torch.nn as nn
from torch.nn import functional as F
from abc import ABC, abstractmethod
from torch.distributions import relaxed_categorical as rc
from torch.distributions import Normal
from torch.autograd import Variable
from utils_distributions import log_Normal_standard, log_Normal_diag
from scipy.stats import chi2
import math
import abc
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(torch.nn.Module):
	def __init__(self, batch_size, input_dim, latent_dim, hidden_dim=50, loss_type='binary', 
			     ordinal_K=None, embedding_dim=5, 
				 z_prior='standard', sigma_prior_scale=1., sigma_prior_df=3, sigmas_init=None,
				 beta_vae=1, genuine_binary_indices=None, categorical_groups=None, **kwargs):
		super(VAE, self).__init__()

		self.batch_size = batch_size
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.latent_dim = latent_dim
		self.sigma_prior_df = sigma_prior_df
		self.sigma_prior_scale = sigma_prior_scale
		self.loss_type = loss_type
		self.ordinal_K = ordinal_K
		self.embedding_dim = embedding_dim
		self.beta_vae=beta_vae
		self.genuine_binary_indices = []
		self.categorical_groups = []

		self.z_prior = z_prior

		if sigmas_init is not None:
			self.log_sigmas = nn.Parameter(torch.log(torch.tensor(sigmas_init, dtype=torch.float)))
		else:
			self.log_sigmas = nn.Parameter(torch.randn(input_dim))

		self.q_z = nn.Sequential(nn.Linear(input_dim, hidden_dim),
								 nn.ReLU(),
								 nn.Linear(hidden_dim, hidden_dim),
								 nn.ReLU())

		self.z_mean = nn.Linear(hidden_dim, latent_dim)
		self.z_log_var = nn.Linear(hidden_dim, latent_dim)

		self.generator = nn.Sequential(nn.Linear(latent_dim, hidden_dim, bias=False),
										nn.ReLU(),
										nn.Linear(hidden_dim, hidden_dim),
										nn.ReLU(),
									    nn.Linear(hidden_dim, input_dim))

		if z_prior == 'covariance':
			self.C = nn.Parameter(torch.randn(self.latent_dim, 2))

		if self.loss_type == 'ordinal':
			# Create a nn.ParameterDict for variable-specific rho parameters
			# each rho is shape (K_j - 1, )
			# construct rhos with len(ordinal_K) list
			self.rhos = nn.ParameterList([
				nn.Parameter(torch.randn(K_j - 1))  # shape: (K_j - 1,)
				for K_j in self.ordinal_K
			])

			# Create embedding layers for each ordinal variable only
			self.embeddings = nn.ModuleList([
				nn.Embedding(K_j+1, self.embedding_dim) # shape: (K_j + 1, embedding_dim)
				for K_j in self.ordinal_K
			])

			self.q_z = nn.Sequential(nn.Linear(input_dim * self.embedding_dim, hidden_dim),
									nn.ReLU(),
									nn.Linear(hidden_dim, hidden_dim),
									nn.ReLU())

		if self.loss_type == 'binary':
			self._configure_binary_features(genuine_binary_indices, categorical_groups)


	def _configure_binary_features(self, genuine_binary_indices, categorical_groups):
		"""Validate the processed-column specification for the mixed discrete VAE."""
		normalized_groups = []
		categorical_index_set = set()

		for group in categorical_groups or []:
			indices = [int(i) for i in group['processed_indices']]
			num_levels = int(group['num_levels'])

			if num_levels < 2:
				raise ValueError("Each categorical variable must have at least two levels.")
			if len(indices) != num_levels - 1:
				raise ValueError(
					"A K-level categorical variable must have exactly K-1 processed columns."
				)
			if len(indices) != len(set(indices)):
				raise ValueError("Categorical processed indices must be unique within a group.")
			if any(i < 0 or i >= self.input_dim for i in indices):
				raise ValueError("Categorical processed index is outside the VAE input dimension.")
			if categorical_index_set.intersection(indices):
				raise ValueError("Categorical processed-column groups must not overlap.")

			categorical_index_set.update(indices)
			normalized_group = dict(group)
			normalized_group['processed_indices'] = indices
			normalized_group['num_levels'] = num_levels
			normalized_groups.append(normalized_group)

		if genuine_binary_indices is None:
			genuine_binary_indices = [
				i for i in range(self.input_dim) if i not in categorical_index_set
			]
		else:
			genuine_binary_indices = [int(i) for i in genuine_binary_indices]

		if len(genuine_binary_indices) != len(set(genuine_binary_indices)):
			raise ValueError("Genuine binary processed indices must be unique.")
		if any(i < 0 or i >= self.input_dim for i in genuine_binary_indices):
			raise ValueError("Genuine binary processed index is outside the VAE input dimension.")
		if categorical_index_set.intersection(genuine_binary_indices):
			raise ValueError("Genuine binary and categorical processed indices must not overlap.")

		covered_indices = categorical_index_set.union(genuine_binary_indices)
		if covered_indices != set(range(self.input_dim)):
			raise ValueError(
				"Every binary-modality input column must be specified as either genuine "
				"binary or part of one categorical group."
			)

		self.genuine_binary_indices = genuine_binary_indices
		self.categorical_groups = normalized_groups


	def _categorical_target(self, x, group):
		"""Convert a K-1 dummy block to class labels in {0, ..., K-1}."""
		group_x = x[:, group['processed_indices']]
		target = torch.argmax(group_x, dim=1).long() + 1
		is_reference = torch.sum(group_x, dim=1) < 0.5
		return torch.where(is_reference, torch.zeros_like(target), target)


	def _categorical_logits(self, x_pred, group):
		"""Prepend the fixed reference-category logit to K-1 learned logits."""
		free_logits = x_pred[:, group['processed_indices']]
		reference_logits = torch.zeros(
			(free_logits.shape[0], 1), dtype=free_logits.dtype, device=free_logits.device
		)
		return torch.cat([reference_logits, free_logits], dim=1)


	def binary_reconstruction_loss_per_sample(self, x_pred, x):
		"""Bernoulli NLL for genuine binaries plus categorical NLL per K-1 block."""
		reconstruction_loss = torch.zeros(
			x.shape[0], dtype=x_pred.dtype, device=x_pred.device
		)

		if self.genuine_binary_indices:
			binary_logits = x_pred[:, self.genuine_binary_indices]
			binary_targets = x[:, self.genuine_binary_indices]
			binary_loss = F.binary_cross_entropy_with_logits(
				binary_logits, binary_targets, reduction='none'
			).sum(dim=1)
			reconstruction_loss = reconstruction_loss + binary_loss

		for group in self.categorical_groups:
			full_logits = self._categorical_logits(x_pred, group)
			target = self._categorical_target(x, group)
			reconstruction_loss = reconstruction_loss + F.cross_entropy(
				full_logits, target, reduction='none'
			)

		return reconstruction_loss


	def binary_reconstruction_probabilities(self, x_pred):
		"""Return probabilities in the unchanged K-1 processed-data layout."""
		probabilities = torch.zeros_like(x_pred)

		if self.genuine_binary_indices:
			probabilities[:, self.genuine_binary_indices] = torch.sigmoid(
				x_pred[:, self.genuine_binary_indices]
			)

		for group in self.categorical_groups:
			full_probabilities = torch.softmax(
				self._categorical_logits(x_pred, group), dim=1
			)
			# The processed representation omits reference category 0.
			probabilities[:, group['processed_indices']] = full_probabilities[:, 1:]

		return probabilities


	def binary_reconstruction_predictions(self, x_pred):
		"""Return hard predictions in the unchanged K-1 processed-data layout."""
		predictions = torch.zeros_like(x_pred)

		if self.genuine_binary_indices:
			binary_probabilities = torch.sigmoid(x_pred[:, self.genuine_binary_indices])
			predictions[:, self.genuine_binary_indices] = (binary_probabilities >= 0.5).to(
				x_pred.dtype
			)

		for group in self.categorical_groups:
			predicted_class = torch.argmax(self._categorical_logits(x_pred, group), dim=1)
			one_hot = F.one_hot(
				predicted_class, num_classes=group['num_levels']
			).to(x_pred.dtype)
			predictions[:, group['processed_indices']] = one_hot[:, 1:]

		return predictions


	def encode(self, x):
		if self.loss_type == 'ordinal':
			
			# Apply embedding for each categorical variable
			embedded_features = [self.embeddings[j](x[:, j]) for j in range(self.input_dim)]

			# Concatenate embeddings
			embedded_features = torch.cat(embedded_features, dim=1)
			# Expected shape: (batch_size, input_dim * embedding_dim)
			
			q_z = self.q_z(embedded_features.float())
			z_mean = self.z_mean(q_z)
			z_log_var = self.z_log_var(q_z)

		else:
			q_z = self.q_z(x)
			z_mean = self.z_mean(q_z)
			z_log_var = self.z_log_var(q_z)

		return z_mean, z_log_var

	def reparameterize(self, mean, log_var):
		std = torch.exp(0.5 * log_var)
		eps = torch.randn_like(std)
		sample = mean + (eps * std)
		return sample

	def decode(self, z):
		x_reconstructed = self.generator(z)
		return x_reconstructed

	def kld_z(self, z, mu, log_var):

		if self.z_prior == 'standard':
			kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

		elif self.z_prior == 'covariance':

			z_cov = torch.matmul(self.C, torch.transpose(self.C, 0, 1)) + torch.eye(self.latent_dim)
			z_cov_inv = torch.inverse(z_cov)

			kld = 0

			for i in range(z.shape[0]):
				kld += torch.matmul(torch.matmul(mu[i, :], z_cov_inv), mu[i, :])
				kld += torch.diagonal(z_cov_inv * log_var[i, :].exp()).sum()

			kld += z.shape[0] * torch.log(torch.det(z_cov))
			kld -= torch.sum(log_var)

			kld = 0.5 * kld / z.shape[0]

		else:
			raise Exception('Wrong name of the prior!')

		return kld

	def rhos_to_thetas(self, rhos):
		thetas = torch.zeros_like(rhos)
		thetas[0] = rhos[0]
		
		for i in range(1, len(rhos)):
			thetas[i] = thetas[i-1] + torch.exp(rhos[i])
			
		return thetas
	
	def ordinal_probit_loss(self, thetas, x_pred, x, K):
		normal_cdf = Normal(0, 1).cdf  # Standard normal CDF

		# Compute the probabilities for each category
		probs = []
		for k in range(1, K + 1):
			upper = normal_cdf(thetas[k - 1] - x_pred) if k < K else 1.0  # Φ(θ_k - w·x)
			lower = normal_cdf(thetas[k - 2] - x_pred) if k > 1 else 0.0  # Φ(θ_{k-1} - w·x)
			probs.append(upper - lower)  # P(y_i = k)

		probs = torch.stack(probs, dim=1)  # Shape: (batch_size, K)

		# Targets x ∈ {1, ..., K} => shift to {0, ..., K-1}
		tmp = torch.log(1e-8 + probs.gather(1, (x - 1).unsqueeze(1).long()))

		log_likelihood = tmp.sum()
		return -log_likelihood, -tmp

	# reconstruction loss
	def reconstruction_loss(self, x_pred, x):
		if self.loss_type == 'mse':
			sigmas = torch.exp(self.log_sigmas)
			loss = nn.MSELoss()
			reconstruction_loss = 0.5 * loss(x_pred / sigmas, (x / sigmas))

		if self.loss_type == 'binary':
			reconstruction_loss = self.binary_reconstruction_loss_per_sample(
				x_pred, x
			).mean()
		
		if self.loss_type == 'ordinal':
			reconstruction_loss = 0
			for j in range(self.input_dim):
				K_j = self.ordinal_K[j]
				rho = self.rhos[j]  # shape: (K_j - 1,)
				thetas = self.rhos_to_thetas(rho)  # shape: (K_j - 1,)
				loss_j = self.ordinal_probit_loss(thetas, x_pred[:, j], x[:, j], K_j)[0]
				reconstruction_loss += loss_j / self.batch_size


		return reconstruction_loss
	
	def reconstruction_loss_outlier(self, x_pred, x, log_sigmas, thetas):
		if self.loss_type == 'mse':
			sigmas = torch.exp(log_sigmas)
			reconstruction_loss  = 0.5*torch.mean((x_pred / sigmas - x / sigmas) ** 2, dim=1)
		
		if self.loss_type == 'binary':
			reconstruction_loss = self.binary_reconstruction_loss_per_sample(x_pred, x)
		
		if self.loss_type == 'ordinal':
			reconstruction_loss = torch.zeros_like(x)
			for j in range(self.input_dim):
				K_j = self.ordinal_K[j]
				tmp = self.ordinal_probit_loss(thetas[:, j], x_pred[:, j], x[:, j], K_j)[1]  # shape: (batch_size, 1)
				reconstruction_loss[:, j] = tmp.squeeze(1)
			# Final per-sample loss by summing across features
			reconstruction_loss = reconstruction_loss.sum(dim=1)

		return reconstruction_loss

	def sigma_loss(self):
		sig_loss = (self.batch_size + self.sigma_prior_df + 2) * self.log_sigmas.sum() \
			+ 0.5 * self.sigma_prior_df * self.sigma_prior_scale * torch.sum(1/torch.exp(2 * self.log_sigmas))

		sig_loss = sig_loss / self.batch_size
		return sig_loss

	def forward(self, x):
		z_mean, z_log_var = self.encode(x)
		z = self.reparameterize(z_mean, z_log_var)
		x_mean = self.decode(z)

		return x_mean, z, z_mean, z_log_var

	def vae_loss(self, x, normalized_x=None):
		if normalized_x is not None:
			x_mean, z, z_mean, z_log_var = self.forward(normalized_x)
		else:
			x_mean, z, z_mean, z_log_var = self.forward(x)

		kl_loss = self.kld_z(z, z_mean, z_log_var)
		reconstruction_loss = self.reconstruction_loss(x_mean, x)

		if self.loss_type == 'mse':
			sigma_loss = self.sigma_loss()
		else:
			sigma_loss = torch.tensor([0.], dtype=torch.float, device=device)

		return reconstruction_loss, kl_loss, sigma_loss


def get_sigma_prior(x):
	# set up sigmas prior
	sigmas_est = x.std(axis=0)
	sig_quant = 0.9
	sig_df = 3

	sig_est = np.quantile(sigmas_est, q=0.05)
	if sig_est==0:
		sig_est = 1e-3

	q_chi = chi2.ppf(1-sig_quant, sig_df)
	sig_scale = sig_est * sig_est * q_chi / sig_df

	sigmas_init = 0.8 * sig_est * np.ones(sigmas_est.shape)
	return sig_scale, sigmas_init

def minmax_scale(loss1, loss2):
	# Min-Max Scaling
	min_loss1 = loss1.min()
	max_loss1 = loss1.max()

	# Scale loss1
	scaled_loss1 = (loss1 - min_loss1) / (max_loss1 - min_loss1)

	# Use the same scale to scale loss2
	scaled_loss2 = (loss2 - min_loss1) / (max_loss1 - min_loss1)

	return scaled_loss1, scaled_loss2


def rhos_to_matrix_numpy(rhos: nn.ParameterList) -> np.ndarray:
  """
  Convert a ParameterList of rhos to a NumPy matrix.

  Args:
      rhos: nn.ParameterList
          Each element is a nn.Parameter of shape (K_j - 1,)

  Returns:
      rho_matrix: np.ndarray of shape (max_K - 1, num_features)
          NaNs are used where a feature has fewer rho entries
  """
  num_features = len(rhos)
  if num_features == 0:
      return np.empty((0, 0), dtype=np.float32)

  # Determine max(K_j - 1)
  max_K_minus_1 = max(r.shape[0] for r in rhos)

  # Initialize matrix with NaNs
  rho_matrix = np.full((max_K_minus_1, num_features), np.nan, dtype=np.float32)

  for j, rho_param in enumerate(rhos):
      rho_np = rho_param.detach().cpu().numpy()
      rho_matrix[:len(rho_np), j] = rho_np

  return rho_matrix

def compute_thetas(rhos):
	rhos = np.asarray(rhos)  # Ensure input is a NumPy array
	m, p = rhos.shape
	thetas = np.full((m, p), np.nan)  # Initialize the output array
	for j in range(p):
		# If first rho is NaN, leave entire column as NaN
		if np.isnan(rhos[0, j]):
			continue
		thetas[0, j] = rhos[0, j]
		for i in range(1, m):
			if np.isnan(rhos[i, j]):
				break  # Subsequent entries remain NaN
			thetas[i, j] = thetas[i - 1, j] + np.exp(rhos[i, j])
	return thetas

def compute_ordinal(y_star, thetas):
	y = np.sum(y_star[:, np.newaxis, :] > thetas[np.newaxis, :, :], axis=1) + 1
	
	return y.astype(int)

def compute_ordinal_quantile(y_star, thetas):
	N, d = y_star.shape
	Km1 = thetas.shape[0]
	Q = np.full((Km1, d), np.nan)  # Pre-fill with NaNs
	for j in range(d):
		sorted_y = np.sort(y_star[:, j])
		for i in range(Km1):
			theta_ij = thetas[i, j]
			if not np.isnan(theta_ij):
				Q[i, j] = np.searchsorted(sorted_y, theta_ij, side='right') / N
	return Q

def compute_ordinal_quantile_true(X, num_classes=None):
    """
    Compute empirical CDF (quantile levels) for ordinal features in X.

    Args:
        X: array-like of shape (N, d)
            Ordinal feature matrix (integer labels per feature)
        num_classes: list or array of int, length d, ordinal_K
            Number of ordinal levels per feature. If None, inferred.

    Returns:
        Q: np.ndarray of shape (max_K - 1, d)
            Cumulative probabilities up to each class (excluding final class).
    """
    X = np.asarray(X)
    N, d = X.shape

    if num_classes is None:
        num_classes = [np.max(X[:, i]) for i in range(d)]

    max_classes = max(num_classes)
    ratio_matrix = np.zeros((max_classes, d))

    for i in range(d):
        unique, counts = np.unique(X[:, i], return_counts=True)
        ratios = counts / N

        # Handle case where classes don't start at 0
        shifted_unique = unique - 1  # Assuming labels are 1-based
        ratio_matrix[shifted_unique, i] = ratios

    # Compute cumulative distribution (quantiles)
    Q = np.cumsum(ratio_matrix, axis=0)[:-1, :]  # exclude final row (P(Y ≤ K) = 1)

    return Q


def plot_histograms(database, split_indices, column='reconstruction_loss', labels=None, bins=20, output_path=None):
    """
    Plots histograms of a specified column from split sections of a dataframe.

    Parameters:
    - database: The full dataframe to be split.
    - split_indices: A list of indices to split the dataframe. Example: [100, 200] will make 3 dfs: [0:100], [100:200], [200:].
    - column: The column name to plot histogram for.
    - labels: Optional list of labels for the splits.
    - bins: Number of bins for the histogram.
    """

    # Generate split DataFrames
    split_dfs = []
    start = 0
    for end in split_indices:
        split_dfs.append(database[start:end])
        start = end
    split_dfs.append(database[start:])  # Final segment

    # Default labels if not provided
    if labels is None:
        labels = [f"Segment {i+1}" for i in range(len(split_dfs))]

    # Plotting
    plt.figure(figsize=(10, 6))
    for df, label in zip(split_dfs, labels):
        plt.hist(df, bins=bins, alpha=0.5, label=label, density=True)

    plt.xlabel(column.replace('_', ' ').title())
    plt.ylabel("Density")
    plt.title("Histogram of Reconstruction Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'Histogram of Reconstruction Losss.png'), dpi=300)
    plt.close()
