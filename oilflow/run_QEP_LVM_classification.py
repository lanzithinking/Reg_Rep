"Q-Exponential Latent Variable Model"

import os
import random
import numpy as np
import matplotlib.pylab as plt
import urllib.request
import tarfile
import timeit

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
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.distributions import MultivariateQExponential

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

POWER = torch.tensor(1.15, device=device)

# define model
def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))

class bQEPLVM(BayesianQEPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing, **kwargs):
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
        X_init = kwargs.pop('X_init', torch.nn.Parameter(torch.randn(n, latent_dim)))

        # LatentVariable (c)
        X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)

        # For (a) or (b) change to below:
        # X = PointLatentVariable(n, latent_dim, X_init)
        # X = MAPLatentVariable(n, latent_dim, X_init, prior_x)

        super().__init__(X, q_f)

        # Kernel (acting on latent dimensions)
        self.mean_module = ZeroMean(ard_num_dims=latent_dim)
        # self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
        self.covar_module = ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=latent_dim))

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateQExponential(mean_x, covar_x, power=self.power)
        return dist

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)


def train_QEPLVM(Y, latent_dim=None, n_inducing=25, batch_size=100, num_epochs=10000, save_model=False, **kwargs):

    N, data_dim = Y.shape
    if latent_dim is None: latent_dim = data_dim
    # n_inducing = 25
    
    # Model
    model = bQEPLVM(N, data_dim, latent_dim, n_inducing, **kwargs)
    # if 'X_init' in kwargs:
    #     delattr(model.X, 'q_mu')
    #     model.X.q_mu = kwargs.pop('X_init')
    if 'X_S_init' in kwargs:
        model.X.q_log_sigma = kwargs.pop('X_S_init')
    
    # Likelihood
    likelihood = QExponentialLikelihood(batch_shape=model.batch_shape, power=torch.tensor(POWER))
    
    # Declaring the objective to be optimised along with optimiser
    # (see models/latent_variable.py for how the additional loss terms are accounted for)
    mll = VariationalELBO(likelihood, model, num_data=N)
    
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=0.004) # 2:0.002, 1.0:0.005, 1.5:0.003
    
    
    loss_list = []
    f_name = 'oilflow_qeplvm_q'+str(POWER.cpu().item())+'_class_'+kwargs.pop('class_','full')+'_checkpoint.dat'
    if os.path.exists(os.path.join('./results',f_name)):
        state_dict = torch.load(os.path.join('./results',f_name), map_location=device)
        # load the best model
        model.load_state_dict(state_dict['model'])
        likelihood.load_state_dict(state_dict['likelihood'])
    else:
        # set device
        model = model.to(device)
        likelihood = likelihood.to(device)
        
        # Training loop - optimises the objective wrt kernel hypers, variational params and inducing inputs
        # using the optimizer provided.
        model.train()
        likelihood.train()
        
        os.makedirs('./results', exist_ok=True)
        # num_epochs = 10000
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
            if 'fix_param' in kwargs:
                fix_param = kwargs.get('fix_param')
                for name, index in fix_param.items():
                    getattr(eval(name), 'grad')[index] = 0
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
        if save_model:
            # save the model
            model_state_dict = optim_model#.state_dict()
            likelihood_state_dict = optim_likelihood#.state_dict()
            torch.save({'model': model_state_dict, 'likelihood': likelihood_state_dict}, os.path.join('./results',f_name))
    # ready for evaluation
    model.eval()
    likelihood.eval()
    
    return model, likelihood, min_loss

def main(seed=2024):
    # Setting manual seed for reproducibility
    # seed=2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    labels = (labels @ np.diag([0, 1, 2])).sum(axis=1)
    batch_size = 100
    num_epochs = 5000
    N, data_dim = Y.shape
    latent_dim = data_dim
    n_class = len(torch.unique(labels))
    
    # split data
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(np.arange(len(Y)), test_size=0.1, random_state=2024)
    n_train, n_test = len(train_idx), len(test_idx)
    Y[train_idx] -= Y[train_idx].mean(0); Y[train_idx] /= Y[train_idx].std(0)
    Y[test_idx] -= Y[train_idx].mean(0); Y[test_idx] /= Y[train_idx].std(0)
    
    log_predprob = torch.zeros((n_class, n_test))
    loss = np.zeros((n_class, 2))
    for k in torch.unique(labels):
        # model based on training data
        Y_train = Y[train_idx][labels[train_idx]==k]
        X_init = _init_pca(Y_train, latent_dim)
        beginning=timeit.default_timer()
        model_train, likelihood_train, loss[k.int()][0] = train_QEPLVM(Y_train, latent_dim, batch_size=batch_size, num_epochs=num_epochs, 
                                                                       save_model=False, class_=str(k.cpu().item())+'train', X_init=X_init)
        time_ = timeit.default_timer()-beginning
        sample = model_train.sample_latent_variable()
        batch_idx = np.random.default_rng(seed).choice(np.arange(Y_train.shape[0]), size=min(batch_size,Y_train.shape[0]), replace=False)
        mll = VariationalELBO(likelihood_train, model_train, num_data=model_train.n)
        elbo_train = mll(model_train(sample[batch_idx]), Y_train[batch_idx].to(device).T).sum()
        # full model
        Y_full = torch.concatenate((Y_train, Y[test_idx])) # appending testing data
        nn_idx = []
        for yte in Y[test_idx]:
            nn_idx.append((Y_train-yte).norm(dim=-1).argmin())
        q_mu_train = model_train.X.q_mu.detach()
        X_init = torch.concatenate((q_mu_train, torch.nn.Parameter(q_mu_train[nn_idx]))) # this will NOT fix q_mu_train
        X_S_init = torch.nn.Parameter(torch.concatenate((model_train.X.q_log_sigma.detach(), torch.randn(n_test, latent_dim, device=device))))
        fix_param = {'model.X.q_mu': np.arange(Y_train.shape[0]),
                     'model.X.q_log_sigma': np.arange(Y_train.shape[0])}
        beginning=timeit.default_timer()
        model_full, likelihood_full, loss[k.int()][1] = train_QEPLVM(Y_full, latent_dim, batch_size=batch_size, num_epochs=num_epochs, 
                                                                     save_model=False, class_=str(k.cpu().item())+'full', X_init=X_init, X_S_init=X_S_init, 
                                                                     fix_param=fix_param)
        time_ += timeit.default_timer()-beginning
        # sample_ = sample.clone()
        sample = model_full.sample_latent_variable()
        # sample[:-n_test] = sample_
        batch_idx = np.concatenate((batch_idx, np.arange(Y_train.shape[0], Y_full.shape[0])))
        loglik = likelihood_full.expected_log_prob(Y_full[batch_idx].to(device).T, model_full(sample[batch_idx]))#.div(len(batch_idx))
        kl = model_full.variational_strategy.kl_divergence().div(model_full.n)
        for added_loss_term in model_full.added_loss_terms():
            kl.add_(added_loss_term.loss())
        elbo_full = loglik[:,-n_test:].sum(0)+loglik[:,:-n_test].sum(0).mean()-kl.sum()
        
        # log prediction probabilities
        log_predprob[k.int()] = elbo_full - elbo_train + torch.log(labels[test_idx].eq(k).float().mean())
    # summarize losses
    import pandas as pd
    loss = pd.DataFrame(loss, columns=['training', 'full'])
    print('Loss:\n'+loss.to_string())
    # make prediction
    from sklearn.metrics import roc_auc_score
    LPP = log_predprob.mean().cpu().item()
    cls_max, pred = torch.max(log_predprob, 0)
    ACC = pred.eq(labels[test_idx].view_as(pred)).float().mean().cpu()
    t_score = torch.exp(log_predprob - cls_max).T if n_class>2 else cls_max
    if t_score.size(1)>1: t_score/=t_score.sum(1,keepdims=True)
    if n_class>2:
        AUC = roc_auc_score(labels[test_idx].cpu(), t_score.detach().cpu(), multi_class='ovo')
    else:
        AUC = roc_auc_score(labels[test_idx].cpu(), t_score.detach().cpu())
    print('Test Accuracy: {}%'.format(100. * ACC))
    print('Test AUC: {}'.format(AUC))
    print('Test LPP: {}'.format(LPP))
    
    # save to file
    os.makedirs('./results', exist_ok=True)
    stats = np.array([ACC, AUC, LPP, time_])
    stats = np.array([seed,'q='+str(POWER.cpu().item())]+[np.array2string(r, precision=4) for r in stats])[None,:]
    header = ['seed', 'Method', 'ACC', 'AUC', 'LPP', 'time']
    f_name = os.path.join('./results','oilflow_latdim_QEP-LVM.txt')
    with open(f_name,'ab') as f:
        np.savetxt(f,stats,fmt="%s",delimiter=',',header=','.join(header) if seed==2024 else '')

if __name__ == '__main__':
    # main()
    n_seed = 10; i=0; n_success=0; n_failure=0
    while n_success < n_seed and n_failure < 10* n_seed:
        seed_i=2024+i*10
        try:
            print("Running for seed %d ...\n"% (seed_i))
            main(seed=seed_i)
            n_success+=1
        except Exception as e:
            print(e)
            n_failure+=1
            pass
        i+=1