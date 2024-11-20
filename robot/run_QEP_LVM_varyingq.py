"Q-Exponential Latent Variable Model"

import os
import random
import numpy as np
import matplotlib.pylab as plt

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
from gpytorch.distributions import MultivariateQExponential, Power

# Setting manual seed for reproducibility
seed=2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

q = 2.0
# POWER = torch.tensor(q, device=device)

# create data
from load_data import *
train, valid, test = load_weights()
Y = torch.tensor(np.vstack((train, valid)))
n_samples = Y.shape[0]
train_dataset = TensorDataset(Y, torch.arange(n_samples))
batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# define model
def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))

class bQEPLVM(BayesianQEPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing, pca=False, power_init=torch.tensor(1.0)):
        super().__init__(None, None)
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        self.power = Power(power_init, power_prior=gpytorch.priors.GammaPrior(4.0, 2.0))

        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)

        # Sparse Variational Formulation (inducing variables initialised as randn)
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape, power=self.power)
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)

        # Define prior for X
        X_prior_mean = torch.zeros(n, latent_dim)  # shape: N x Q
        prior_x = QExponentialPrior(X_prior_mean, torch.ones_like(X_prior_mean))#, power=self.power.data)
        prior_x.power = self.power

        # Initialise X with PCA or randn
        if pca == True:
             X_init = _init_pca(Y.float(), latent_dim) # Initialise X to PCA
        else:
             X_init = torch.nn.Parameter(torch.randn(n, latent_dim))

        # LatentVariable (c)
        X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x, power=self.power)

        # For (a) or (b) change to below:
        # X = PointLatentVariable(n, latent_dim, X_init)
        # X = MAPLatentVariable(n, latent_dim, X_init, prior_x)

        # super().__init__(X, q_f)
        self.X = X
        self.variational_strategy = q_f

        # Kernel (acting on latent dimensions)
        self.mean_module = ZeroMean(ard_num_dims=latent_dim)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
        
        # define likelihood
        self.likelihood = QExponentialLikelihood(batch_shape=self.batch_shape, power=self.power)

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
latent_dim = 10
n_inducing = 25
pca = False
power_init = torch.tensor(q)

# Model
model = bQEPLVM(N, data_dim, latent_dim, n_inducing, pca=pca, power_init=power_init)

# Likelihood
# likelihood = QExponentialLikelihood(batch_shape=model.batch_shape, power=torch.tensor(POWER))

# Declaring the objective to be optimised along with optimiser
# (see models/latent_variable.py for how the additional loss terms are accounted for)
mll = VariationalELBO(model.likelihood, model, num_data=len(Y))

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    # {'params': likelihood.parameters()}
], lr=0.01)


loss_list = []
if os.path.exists(os.path.join('./results','robot_qeplvm_varyingq_checkpoint.dat')):
    state_dict = torch.load(os.path.join('./results','robot_qeplvm_varyingq_checkpoint.dat'), map_location=device)['model']
else:
    # set device
    model = model.to(device)
    # likelihood = likelihood.to(device)
    # mll = mll.to(device)
    
    # Training loop - optimises the objective wrt kernel hypers, variational params and inducing inputs
    # using the optimizer provided.
    model.train()
    # likelihood.train()
    
    os.makedirs('./results', exist_ok=True)
    # loss_list = []
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
            # loss_list.append(loss.item())
            # iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
        loss.backward()
        optimizer.step()
            # minibatch_iter.set_postfix(loss=loss.item())
        loss_list.append(loss.item())
        if epoch==0:
            min_loss = loss_list[-1]
            optim_model = model.state_dict()
            optim_likelihood = model.likelihood.state_dict()
        else:
            if loss_list[-1] < min_loss:
                min_loss = loss_list[-1]
                optim_model = model.state_dict()
                optim_likelihood = model.likelihood.state_dict()
        print('Epoch {}/{}: Loss: {}, power: {}'.format(epoch, num_epochs, loss.item(), model.power.data.item()))
    # save the model
    state_dict = optim_model#.state_dict()
    likelihood_state_dict = optim_likelihood#.state_dict()
    torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, os.path.join('./results','robot_qeplvm_varyingq_checkpoint.dat'))

# load the best model
model.load_state_dict(state_dict)
model.eval()
# plot results
inv_lengthscale = 1 / model.covar_module.base_kernel.lengthscale
values, indices = torch.topk(model.covar_module.base_kernel.lengthscale, k=2,largest=False)
l1, l2 = indices.detach().cpu().numpy().flatten()[:2]
q = round(model.power.cpu().data.item(), 2)

idx2plot = model._get_batch_idx(500, seed)
X = model.X.q_mu.detach().cpu().numpy()[idx2plot]

# plot
plt.figure(figsize=(20, 6))
plt.subplot(131)
plt.scatter(X[:, l1], X[:, l2], alpha=0.8)
# plt.xlim([-1,1]); plt.ylim([-1,1])
plt.title('2d latent subspace', fontsize=20)
plt.xlabel('Latent dim 1', fontsize=20)
plt.ylabel('Latent dim 2', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.subplot(132)
plt.bar(np.arange(latent_dim), height=inv_lengthscale.detach().cpu().numpy().flatten())
plt.title('Inverse Lengthscale of kernel', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.subplot(133)
plt.plot(loss_list, label='batch_size='+str(batch_size))
plt.title('Neg. ELBO Loss', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)
# plt.show()
os.makedirs('./results', exist_ok=True)
plt.savefig(os.path.join('./results','robot_QEP-LVM_q'+str(q)+'.png'),bbox_inches='tight')

fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, l1], X[:, l2], alpha=0.8)
plt.title('q = '+str(q)+(' (Gaussian)' if q==2 else ''), fontsize=20)
# plt.axis('square')
plt.xlabel('Latent dim 1', fontsize=18)
plt.ylabel('Latent dim 2', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join('./results','robot_latent_QEP-LVM_q'+str(q)+'.png'),bbox_inches='tight')

fig = plt.figure(figsize=(8, 6))
plt.bar(np.arange(latent_dim), height=inv_lengthscale.detach().cpu().numpy().flatten())
plt.title('Inverse Lengthscale of kernel', fontsize=20)
plt.ylabel(' ', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join('./results','robot_latdim_QEP-LVM_q'+str(q)+'.png'),bbox_inches='tight')

# plot latent space
fig = plt.figure(figsize=(8,6))
from sklearn.manifold import TSNE
model_tsne = TSNE(n_components=2, random_state=seed)
Z_ = model_tsne.fit_transform(X)
plt.scatter(Z_[:, 0], Z_[:, 1])
plt.xlabel('Latent dim 1', fontsize=20)
plt.ylabel('Latent dim 2', fontsize=20)
plt.savefig(os.path.join('./results','robot_QEP-LVM_q'+str(q)+'_latent.png'),bbox_inches='tight')

# plot densities
if 'sample' not in globals():
    sample = model.sample_latent_variable()
qs = np.linspace(0, 4, num=50, endpoint=False)
qs += qs[1]
from sklearn.metrics import auc
def likelihood(qs, normalize=True):
    ps = np.zeros_like(qs)
    for i,q in enumerate(qs):
        model.power.power = q
        # ps[i] = model.likelihood.log_marginal(Y.to(device).T, model(sample)).mean()
        ps[i] = model.likelihood.expected_log_prob(Y.to(device).T, model(sample)).mean()
    ps = np.exp(ps-ps.max())
    if normalize:
        area = auc(qs, ps)
        ps /= area
    return ps
def prior(qs, normalize=True):
    ps = model.power.power_prior.log_prob(qs)
    ps = np.exp(ps-ps.max())
    if normalize:
        area = auc(qs, ps)
        ps /= area
    return ps
def posterior(qs, normalize=True):
    ps = np.zeros_like(qs)
    for i,q in enumerate(qs):
        model.power.power = q
        ps[i] = mll(model(sample), Y.to(device).T).mean()
    ps = np.exp(ps-ps.max())
    if normalize:
        area = auc(qs, ps)
        ps /= area
    return ps
fig = plt.figure(figsize=(7, 6))
plt.plot(qs, likelihood(qs))
plt.plot(qs, prior(qs))
# plt.plot(qs, posterior(qs))
plt.axvline(q, linewidth=3, color='red')
plt.legend(labels=['likelihood', 'prior', 'MAP'], fontsize=14, frameon=False)
# plt.legend(labels=['prior', 'posterior', 'MAP'], fontsize=14, frameon=False)
plt.title('Posterior', fontsize=20)
plt.xlabel('q', fontsize=18)
plt.ylabel(' ', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join('./results','robot_densities_QEP-LVM_q'+str(q)+'.png'),bbox_inches='tight')