"Q-Exponential Latent Variable Model"

import os
import random
import numpy as np
import matplotlib.pylab as plt

from sklearn.datasets import make_swiss_roll
import torch
from torch.utils.data import TensorDataset, DataLoader
import tqdm

# gpytorch imports
import sys
sys.path.insert(0,'../GPyTorch')
import gpytorch
from gpytorch.models.qeplvm.latent_variable import *
from gpytorch.models.qeplvm.bayesian_qeplvm import BayesianQEPLVM
from gpytorch.means import ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import QExponentialPrior
from gpytorch.likelihoods import QExponentialLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateQExponential

# Setting manual seed for reproducibility
seed=2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

q = 2.5
POWER = torch.tensor(q, device=device)

# create data
n_samples = 1000
sr_points, sr_color = make_swiss_roll(n_samples=n_samples, noise=0.05, random_state=0)
Y, t = torch.tensor(sr_points), torch.tensor(sr_color)
train_dataset = TensorDataset(Y, t, torch.arange(n_samples))
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection="3d")
# fig.add_axes(ax)
# ax.scatter(
#     sr_points[:, 0], sr_points[:, 1], sr_points[:, 2], c=sr_color, alpha=0.8
# )
# # ax.set_title("Swiss Roll", fontsize=18)# in Ambient Space")
# ax.view_init(azim=-66, elev=12)
# os.makedirs('./results', exist_ok=True)
# plt.savefig(os.path.join('./results','SwissRoll.png'),bbox_inches='tight')

# define model
def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))


class bQEPLVM(BayesianQEPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing, pca=False):
        self.power = torch.tensor(POWER)
        self.n = n
        self.batch_shape = torch.Size([data_dim])

        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)

        # Sparse Variational Formulation (inducing variables initialised as randn)
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape, power=self.power)
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)

        # Define prior for X
        X_prior_mean = torch.zeros(n, latent_dim)  # shape: N x Q
        prior_x = QExponentialPrior(X_prior_mean, torch.ones_like(X_prior_mean), power=self.power)

        # Initialise X with PCA or randn
        if pca == True:
             X_init = _init_pca(Y.float(), latent_dim) # Initialise X to PCA
        else:
             X_init = torch.nn.Parameter(torch.randn(n, latent_dim))

        # LatentVariable (c)
        X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)

        # For (a) or (b) change to below:
        # X = PointLatentVariable(n, latent_dim, X_init)
        # X = MAPLatentVariable(n, latent_dim, X_init, prior_x)

        super().__init__(X, q_f)

        # Kernel (acting on latent dimensions)
        self.mean_module = ZeroMean(ard_num_dims=latent_dim)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateQExponential(mean_x, covar_x, power=self.power)
        return dist

    def _get_batch_idx(self, batch_size, seed=None):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False) if seed is None else \
                        np.random.default_rng(seed).choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)


N = len(Y)
data_dim = Y.shape[1]
latent_dim = data_dim
n_inducing = 25
pca = False

# Model
model = bQEPLVM(N, data_dim, latent_dim, n_inducing, pca=pca)

# Likelihood
likelihood = QExponentialLikelihood(batch_shape=model.batch_shape, power=torch.tensor(POWER))

# Declaring the objective to be optimised along with optimiser
# (see models/latent_variable.py for how the additional loss terms are accounted for)
mll = VariationalELBO(likelihood, model, num_data=len(Y))

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()}
], lr=0.01)

# set device
model = model.to(device)
likelihood = likelihood.to(device)
# mll = mll.to(device)

# Training loop - optimises the objective wrt kernel hypers, variational params and inducing inputs
# using the optimizer provided.

loss_list = []
num_epochs = 10000
iterator = tqdm.tqdm(range(num_epochs), desc="Epoch")
# batch_size = 256
for epoch in iterator:
    batch_index = model._get_batch_idx(batch_size)
    optimizer.zero_grad()
    sample = model.sample_latent_variable()  # a full sample returns latent x across all N
    sample_batch = sample[batch_index]
    output_batch = model(sample_batch)
    loss = -mll(output_batch, Y[batch_index].to(device).T).sum()
    # minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
    # for data, target, batch_index in minibatch_iter:
    #     if torch.cuda.is_available():
    #         data = data.cuda()
    #     optimizer.zero_grad()
    #     sample = model.sample_latent_variable()
    #     output_batch = model(sample[batch_index])
    #     loss = -mll(output_batch, data.flatten(1).T).sum()
    #     # loss_list.append(loss.item())
    #     # iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
    loss.backward()
    optimizer.step()
        # minibatch_iter.set_postfix(loss=loss.item())
    loss_list.append(loss.item())
    print('Epoch {}/{}: Loss: {}'.format(epoch, num_epochs, loss.item() ))


# plot results

inv_lengthscale = 1 / model.covar_module.base_kernel.lengthscale
values, indices = torch.topk(model.covar_module.base_kernel.lengthscale, k=2,largest=False)

l1, l2 = indices.detach().cpu().numpy().flatten()[:2]

idx2plot = model._get_batch_idx(500, seed)
X = model.X.q_mu.detach().cpu().numpy()[idx2plot]
colors = t[idx2plot]

# plt.figure(figsize=(20, 6))
# plt.subplot(131)
# # std = torch.nn.functional.softplus(model.X.q_log_sigma).detach().numpy()
# # Select index of the smallest lengthscales by examining model.covar_module.base_kernel.lengthscales
# # for i, label in enumerate(np.unique(labels)):
# #     X_i = X[labels == label]
# #     # scale_i = std[labels == label]
# #     # plt.scatter(X_i[:, l1], X_i[:, l2], c=[colors[i]], label=label)
# #     plt.scatter(X_i[:, l1], X_i[:, l2], c=[colors[i]], marker="$"+str(label)+"$")
# #     # plt.errorbar(X_i[:, l1], X_i[:, l2], xerr=scale_i[:,l1], yerr=scale_i[:,l2], label=label,c=colors[i], fmt='none')
# plt.scatter(X[:, l1], X[:, l2], c=colors, alpha=0.8)
# # plt.xlim([-1,1]); plt.ylim([-1,1])
# plt.title('2d latent subspace', fontsize=20)
# plt.xlabel('Latent dim 1', fontsize=20)
# plt.ylabel('Latent dim 2', fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=14)
#
# plt.subplot(132)
# plt.bar(np.arange(latent_dim), height=inv_lengthscale.detach().cpu().numpy().flatten())
# plt.title('Inverse Lengthscale of SE-ARD kernel', fontsize=18)
# plt.tick_params(axis='both', which='major', labelsize=14)
#
# plt.subplot(133)
# plt.plot(loss_list, label='batch_size='+str(batch_size))
# plt.title('Neg. ELBO Loss', fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=14)
# # plt.show()
# os.makedirs('./results', exist_ok=True)
# plt.savefig(os.path.join('./results','swissroll_QEP-LVM_q'+str(q)+'.png'),bbox_inches='tight')
#
# fig = plt.figure(figsize=(8, 6))
# plt.scatter(X[:, l1], X[:, l2], c=colors, alpha=0.8)
# plt.title('q = '+str(q)+(' (Gaussian)' if q==2 else ''), fontsize=20)
# plt.axis('square')
# plt.xlabel('Latent dim 1', fontsize=18)
# plt.ylabel('Latent dim 2', fontsize=18)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.savefig(os.path.join('./results','swissroll_latent_QEP-LVM_q'+str(q)+'.png'),bbox_inches='tight')

fig = plt.figure(figsize=(8, 6))
plt.bar(np.arange(latent_dim), height=inv_lengthscale.detach().cpu().numpy().flatten())
plt.title('Inverse Lengthscale of kernel', fontsize=20)
plt.ylabel(' ', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join('./results','swissroll_latdim_QEP-LVM_q'+str(q)+'.png'),bbox_inches='tight')