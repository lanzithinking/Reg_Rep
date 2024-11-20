"To plot pairwise density"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from joblib import Parallel, delayed
import multiprocessing

# define marginal density plot
def plot_pdf(x, **kwargs):
    nx = len(x)
    # z = np.zeros(nx)
    para0 = kwargs.pop('para0',None) # DataFrame
    f = kwargs.pop('f',None)
    # for i in range(nx):
    def parfor(i):
        para_ = para0.copy()
        para_[x.name] = x[i]
        # z[i] = f(list(para_.values()))
        # return f(list(para_.values()))
        return f(*para_.values)
    n_jobs = np.min([1, multiprocessing.cpu_count()])
    z = Parallel(n_jobs=n_jobs)(delayed(parfor)(i) for i in range(nx))
    # z = np.array(z)
    z = torch.stack(z).detach().numpy()
    
    plt.plot(x, z, **kwargs)
    plt.axvline(x=para0[x.name].item(),color='red',linestyle='--',linewidth=1.5)

# define contour function
def contour(x, y, **kwargs):
    nx = len(x); ny = len(y)
    # z = np.zeros((nx, ny))
    para0 = kwargs.pop('para0',None) # DataFrame
    f = kwargs.pop('f',None)
    # for i in range(nx):
        # for j in range(ny):
    def parfor(i, j):
        para_ = para0.copy()
        para_[x.name] = x[i]; para_[y.name] = y[j]
            # z[i,j] = f(list(para_.values()))
        # return f(list(para_.values()))
        return f(*para_.values)
    n_jobs = np.min([10, multiprocessing.cpu_count()])
    z = Parallel(n_jobs=n_jobs)(delayed(parfor)(i,j) for i in range(nx) for j in range(ny))
    # z = np.array(z).reshape(nx,ny)
    z = torch.stack(z).detach().numpy().reshape(nx,ny)
    
    # plt.contourf(x, y, z, **kwargs)
    plt.contourf(x, y, z, levels=np.quantile(z,[.67,.9,.99]), **kwargs)
    plt.axvline(x=para0[x.name].item(),color='red',linestyle='--',linewidth=1.5)
    plt.axhline(y=para0[y.name].item(),color='red',linestyle='--',linewidth=1.5)


if __name__=='__main__':
    import os
    import random
    
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
    from gpytorch.distributions import MultivariateQExponential, Power
    
    # set up random seed
    seed=2021
    np.random.seed(seed)
    
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
    dataset = {0:'swissroll',1:'swisshole'}[0]
    n_samples = 1000
    sr_points, sr_color = make_swiss_roll(n_samples=n_samples, noise=0.05, random_state=0, hole='hole' in dataset)
    Y, t = torch.tensor(sr_points), torch.tensor(sr_color)
    train_dataset = TensorDataset(Y, t, torch.arange(n_samples))
    batch_size = 256
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
                 X_init = torch.nn.Parameter(torch.rand(n, latent_dim))
    
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
    latent_dim = data_dim
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
    if os.path.exists(os.path.join('./results_'+dataset,dataset+'_qeplvm_varyingq_checkpoint.dat')):
        state_dict = torch.load(os.path.join('./results_'+dataset,dataset+'_qeplvm_varyingq_checkpoint.dat'), map_location=device)['model']
    else:
        # set device
        model = model.to(device)
        # likelihood = likelihood.to(device)
        # mll = mll.to(device)
        
        # Training loop - optimises the objective wrt kernel hypers, variational params and inducing inputs
        # using the optimizer provided.
        model.train()
        # likelihood.train()
        
        os.makedirs('./results_'+dataset, exist_ok=True)
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
        torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, os.path.join('./results_'+dataset,dataset+'_qeplvm_varyingq_checkpoint.dat'))
    
    # load the best model
    model.load_state_dict(state_dict)
    model.eval()
    q = round(model.power.cpu().data.item(), 2)
    
    # density
    if 'sample' not in globals():
        sample = model.sample_latent_variable()
    def logpost(param):
        model.power.power = param[0]
        model.covar_module.base_kernel.lengthscale = param[1:]
        return mll(model(sample), Y.to(device).T).mean()
    
    # prepare for plotting data
    para0 = torch.cat((model.power.data[None], model.covar_module.base_kernel.lengthscale.squeeze()))
    marg = torch.ones(len(para0)); res = 100
    grid_data = torch.empty((len(para0),res))
    for i,m in enumerate(marg):
        grid_data[i] = torch.linspace(para0[i]-m, para0[i]+m, steps=res)
    grid_data = pd.DataFrame(grid_data.T,columns=['q','$l_1$','$l_2$', '$l_3$'])
    
    # plot
    os.makedirs('./results_'+dataset, exist_ok=True)
    sns.set(font_scale=1.2)
    import time
    t_start=time.time()
    g = sns.PairGrid(grid_data, diag_sharey=False, corner=True, height=len(para0))#, size=3)
    g.map_diag(plot_pdf, para0=pd.DataFrame(para0.detach().numpy()[None,:], columns=grid_data.columns), f=logpost)
    g.map_lower(contour, para0=pd.DataFrame(para0.detach().numpy()[None,:], columns=grid_data.columns), f=logpost, cmap='gray')
    for ax in g.axes.flatten():
        if ax:
            # # rotate x axis labels
            # if ax.get_xlabel()!='': ax.set_xlabel('$\\'+ax.get_xlabel()+'$')
            # rotate y axis labels
            if ax.get_ylabel()!='': #ax.set_ylabel('$\\'+ax.get_ylabel()+'$', rotation = 0)
                ax.set_ylabel(ax.get_ylabel(), rotation = 0)
            # set y labels alignment
            ax.yaxis.get_label().set_horizontalalignment('right')
    g.savefig(os.path.join('./results_'+dataset, dataset+'_pairpdf_QEP-LVM_q'+str(q)+'.png'),bbox_inches='tight')
    t_end=time.time()
    print('time used: %.5f'% (t_end-t_start))