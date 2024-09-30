"Q-Exponential Latent Variable Model"

import os
import random
import numpy as np
import matplotlib.pylab as plt
import urllib.request
import tarfile

import torch
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

POWER = torch.tensor(1.0, device=device)

# load data
if not (os.path.exists('DataTrn.txt') and os.path.exists('DataTrnLbls.txt')):
    # download data
    url = "http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/resources/3PhData.tar.gz"
    urllib.request.urlretrieve(url, '3PhData.tar.gz')
    with tarfile.open('3PhData.tar.gz', 'r') as f:
        f.extract('DataTrn.txt')
        f.extract('DataTrnLbls.txt')

Y = torch.Tensor(np.loadtxt(fname='DataTrn.txt'))
labels = torch.Tensor(np.loadtxt(fname='DataTrnLbls.txt'))
labels = (labels @ np.diag([1, 2, 3])).sum(axis=1)
batch_size = 100

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
             X_init = _init_pca(Y, latent_dim) # Initialise X to PCA
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

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
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


loss_list = []
if os.path.exists(os.path.join('./results','qeplvm_q'+str(POWER.cpu().item())+'_oilflow_checkpoint.dat')):
    state_dict = torch.load(os.path.join('./results','qeplvm_q'+str(POWER.cpu().item())+'_oilflow_checkpoint.dat'), map_location=device)['model']
else:
    # set device
    model = model.to(device)
    likelihood = likelihood.to(device)
    
    # Training loop - optimises the objective wrt kernel hypers, variational params and inducing inputs
    # using the optimizer provided.
    model.train()
    likelihood.train()
    
    os.makedirs('./results', exist_ok=True)
    num_epochs = 10000
    iterator = tqdm.tqdm(range(num_epochs), desc="Epoch")
    for epoch in iterator:
        batch_index = model._get_batch_idx(batch_size)
        optimizer.zero_grad()
        sample = model.sample_latent_variable()  # a full sample returns latent x across all N
        sample_batch = sample[batch_index]
        output_batch = model(sample_batch)
        loss = -mll(output_batch, Y[batch_index].to(device).T).sum()
        # iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        if epoch==0:
            min_loss = loss_list[-1]
            optim_model = model.state_dict()
            optim_likelihood = likelihood.state_dict()
        else:
            if loss_list[-1] < min_loss:
                min_loss = loss_list[-1]
                optim_model = model.state_dict()
                optim_likelihood = likelihood.state_dict()
        print('Epoch {}/{}: Loss: {}'.format(epoch, num_epochs, loss.item() ))
    # save the model
    state_dict = optim_model#.state_dict()
    likelihood_state_dict = optim_likelihood#.state_dict()
    torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, os.path.join('./results','qeplvm_q'+str(POWER.cpu().item())+'_oilflow_checkpoint.dat'))

# load the best model
model.load_state_dict(state_dict)
model.eval()

# plot results
inv_lengthscale = 1 / model.covar_module.base_kernel.lengthscale
values, indices = torch.topk(model.covar_module.base_kernel.lengthscale, k=2,largest=False)
l1, l2 = indices.detach().cpu().numpy().flatten()[:2]
X = model.X.q_mu.detach().cpu().numpy()
std = torch.nn.functional.softplus(model.X.q_log_sigma).cpu().detach().numpy()
labels = labels.numpy()

# plot
import matplotlib.colors as mcolors
colors = list(mcolors.TABLEAU_COLORS.values())

# plt.figure(figsize=(20, 6))
# plt.subplot(131)
# # Select index of the smallest lengthscales by examining model.covar_module.base_kernel.lengthscales
# for i, label in enumerate(np.unique(labels)):
#     X_i = X[labels == label]
#     scale_i = std[labels == label]
#     plt.scatter(X_i[:, l1], X_i[:, l2], c=[colors[i]], label=label)
#     plt.errorbar(X_i[:, l1], X_i[:, l2], xerr=scale_i[:,l1], yerr=scale_i[:,l2], label=label,c=colors[i], fmt='none')
# plt.title('2d latent subspace', fontsize=20)# corresponding to 3 phase oilflow')
# plt.xlabel('Latent dim 1', fontsize=20)
# plt.ylabel('Latent dim 2', fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.subplot(132)
# plt.bar(np.arange(latent_dim), height=inv_lengthscale.detach().cpu().numpy().flatten())
# plt.title('Inverse Lengthscale of kernel', fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.subplot(133)
# plt.plot(loss_list, label='batch_size=100')
# plt.title('Neg. ELBO Loss', fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=14)
# # plt.show()
# plt.savefig(os.path.join('./results','./oilflow_QEP-LVM_q'+str(POWER.cpu().item())+'.png'),bbox_inches='tight')

plt.figure(figsize=(7, 6))
# plt.contourf(X[:,l1], X[:,l2], 1/np.maximum(std[:,l1], std[:,[l2]]), cmap='gray', alpha=0.5)
plt.contourf(X[:,l1], X[:,l2], np.sqrt(std[:,[l1]]**2 + std[:,l2]**2), cmap='gray', alpha=0.5)
for i, label in enumerate(np.unique(labels)):
    X_i = X[labels == label]
    scale_i = std[labels == label]
    plt.scatter(X_i[:, l1], X_i[:, l2], c=[colors[i]], label=label)
# import pandas as pd
# import seaborn as sns
# dat2plot = pd.DataFrame(np.hstack((X[:,[l1,l2]],std[:,[l1,l2]], labels[:,None])),columns=['latdim_'+str(j) for j in range(2)]+['stddim_'+str(j) for j in range(2)]+['label'])
# dat2plot['label']=dat2plot['label'].astype(int)
# sns.relplot(data=dat2plot, x='latdim_0', y='latdim_1', hue='label', style='label', palette=colors[:len(np.unique(labels))], legend=False)
plt.title('q = '+str(POWER.cpu().item())+(' (Gaussian)' if POWER.cpu().item()==2 else ''), fontsize=20)
plt.xlabel('Latent dim 1', fontsize=20)
plt.ylabel('Latent dim 2', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join('./results','./oilflow_latent_QEP-LVM_q'+str(POWER.cpu().item())+'.png'),bbox_inches='tight')

plt.figure(figsize=(7, 6))
plt.bar(np.arange(latent_dim), height=inv_lengthscale.detach().cpu().numpy().flatten())
plt.title('Inverse Lengthscale of kernel', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join('./results','./oilflow_latdim_QEP-LVM_q'+str(POWER.cpu().item())+'.png'),bbox_inches='tight')