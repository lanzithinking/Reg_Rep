"Deep Gaussian Process Model"

import os
import random
import numpy as np
from matplotlib import pyplot as plt

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tqdm

# gpytorch imports
import sys
sys.path.insert(0,'../GPyTorch')
import gpytorch
from gpytorch.means import ZeroMean, ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.gplvm.latent_variable import *
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.priors import NormalPrior
from gpytorch.likelihoods import MultitaskGaussianLikelihood

# Setting manual seed for reproducibility
seed=2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using the '+device+' device...')

# load data
def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/19
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])
# train_dataset = datasets.MNIST('./data/MNIST', train=True, download=True, transform=transform)
train_dataset = dataset_with_indices(datasets.MNIST)('./data/MNIST', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data/MNIST', train=False, download=True, transform=transform)

Y = train_dataset.data.flatten(1).float()
labels = train_dataset.targets
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# define model
def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))

# Here's a simple standard layer
class DGPLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='zero', **kwargs):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy, input_dims, output_dims)
        n = kwargs.pop('n',1); 
        # self.mean_module = ConstantMean() if not linear_mean else LinearMean(n)#input_dims)
        self.mean_module = {'zero': ZeroMean(ard_num_dims=input_dims),
                            'constant': ConstantMean(),
                            'linear': LinearMean(input_dims)}[mean_type]
        # self.covar_module = ScaleKernel(
        #     RBFKernel(
        #         lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
        #             np.exp(-1), np.exp(1), sigma=0.1, transform=torch.exp
        #         ), batch_shape=batch_shape, ard_num_dims=input_dims
        #     )
        # )
        self.covar_module = ScaleKernel(
            MaternKernel(nu=1.5, batch_shape=batch_shape, ard_num_dims=input_dims,
                # lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                #     np.exp(-1), np.exp(1), sigma=0.1, transform=torch.exp)
            ),
            batch_shape=batch_shape,
        )
        
        # LatentVariable (c)
        data_dim = output_dims; latent_dim = input_dims
        latent_prior_mean = torch.zeros(n, latent_dim)
        latent_prior = NormalPrior(latent_prior_mean, torch.ones_like(latent_prior_mean))
        latent_init = kwargs.pop('latent_init', None)
        if latent_init is not None:
            self.latent_variable = VariationalLatentVariable(n, data_dim, latent_dim, latent_init, latent_prior)

        # For (a) or (b) change to below:
        # X = PointLatentVariable(n, latent_dim, latent_init)
        # X = MAPLatentVariable(n, latent_dim, latent_init, latent_prior)

    def forward(self, x, projection=None):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# define the main model
class MultitaskDeepGP(DeepGP):
    def __init__(self, n, in_features, out_features, hidden_features=2, latent_init=None):
        self.n = n
        super().__init__()
        if isinstance(hidden_features, int):
            layer_config = torch.cat([torch.arange(in_features, out_features, step=(out_features-in_features)/max(1,hidden_features)).type(torch.int), torch.tensor([out_features])])
        elif isinstance(hidden_features, list):
            layer_config = [in_features]+hidden_features+[out_features]
        layers = []
        for i in range(len(layer_config)-1):
            layers.append(DGPLayer(
                input_dims=layer_config[i],
                output_dims=layer_config[i+1],
                mean_type='zero' if i==0 else 'linear' if i < len(layer_config)-2 else 'constant',
                n=n if i==0 else layer_config[i],
                latent_init=latent_init if i==0 else None
            ))
        self.num_layers = len(layers)
        self.layers = torch.nn.Sequential(*layers)

        # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=out_features)

    def forward(self, inputs):
        output = self.layers[0](inputs)#, are_samples=True)
        for i in range(1,len(self.layers)):
            output = self.layers[i](output)
        return output
    
    def _get_batch_idx(self, batch_size, seed=None):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False) if seed is None else \
                        np.random.default_rng(seed).choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)

N, data_dim = Y.shape
latent_dim = 10
hidden_features = [latent_dim]
pca = False
latent_init = _init_pca(Y.float(), latent_dim) if pca else torch.nn.Parameter(torch.randn(N, latent_dim))
model = MultitaskDeepGP(N, latent_dim, data_dim, hidden_features, latent_init)

# training
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=N))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loss_list = []
if os.path.exists(os.path.join('./results','dgp_'+str(model.num_layers)+'layers_mnist_checkpoint.dat')):
    state_dict = torch.load(os.path.join('./results','dgp_'+str(model.num_layers)+'layers_mnist_checkpoint.dat'), map_location=device)['model']
else:
    # set device
    model = model.to(device)
    # mll = mll.to(device)
    
    model.train()
    
    os.makedirs('./results', exist_ok=True)
    # loss_list = []
    num_epochs = 10000
    iterator = tqdm.tqdm(range(num_epochs), desc="Epoch", file=open(os.devnull, 'w'))
    for epoch in iterator:
        batch_index = model._get_batch_idx(batch_size)
        optimizer.zero_grad()
        sample = model.layers[0].latent_variable()
        if sample.ndim==1: sample = sample.unsqueeze(0)
        output_batch = model(sample[batch_index])
        loss = -mll(output_batch, Y[batch_index].to(device)).sum()
        loss.backward()
        optimizer.step()
        # minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
        # for data, target, batch_index in minibatch_iter:
        #     if torch.cuda.is_available():
        #         data = data.cuda()
        #     optimizer.zero_grad()
        #     sample = model.layers[0].latent_variable()
        #     output_batch = model(sample[batch_index])
        #     loss = -mll(output_batch, data.flatten(1).T).sum()
        #     loss.backward()
        #     optimizer.step()
        #     minibatch_iter.set_postfix(loss=loss.item())
        # record the loss and the best model
        loss_list.append(loss.item())
        if epoch==0:
            min_loss = loss_list[-1]
            optim_model = model.state_dict()
        else:
            if loss_list[-1] < min_loss:
                min_loss = loss_list[-1]
                optim_model = model.state_dict()
        print('Epoch {}/{}: Loss: {}'.format(epoch, num_epochs, loss.item() ))
    # save the model
    state_dict = optim_model#.state_dict()
    torch.save({'model': state_dict}, os.path.join('./results','dgp_'+str(model.num_layers)+'layers_mnist_checkpoint.dat'))

# load the best model
model.load_state_dict(state_dict)
model.eval()
# plot results
inv_lengthscale = 1 / model.layers[0].covar_module.base_kernel.lengthscale.mean(0)
values, indices = torch.topk(model.layers[0].covar_module.base_kernel.lengthscale.mean(0), k=2,largest=False)
l1, l2 = indices.detach().cpu().numpy().flatten()[:2]
X = getattr(model.layers[0].latent_variable, 'q_mu' if isinstance(model.layers[0].latent_variable, VariationalLatentVariable) else 'X').detach().cpu().numpy()
labels = labels.numpy()

# plot
import matplotlib.colors as mcolors
colors = list(mcolors.TABLEAU_COLORS.values())

# plt.figure(figsize=(20, 6))
# # idx2plot = model._get_batch_idx(500, seed)
# cls2plot = np.unique(labels)
# num_pcls = 20
# idx2plot = []
# for c in cls2plot:
#     idx2plot.append(np.random.default_rng(seed).choice(np.where(labels==c)[0], size=num_pcls, replace=False))
# idx2plot = np.concatenate(idx2plot)
# X_ = X[idx2plot]
# labels_ = labels[idx2plot]
#
# plt.subplot(131)
# for i, label in enumerate(np.unique(labels_)):
#     X_i = X_[labels_ == label]
#     plt.scatter(X_i[:, l1], X_i[:, l2], c=[colors[i]], marker="$"+str(label)+"$")
# plt.title('2d latent subspace', fontsize=20)
# plt.xlabel('Latent dim 1', fontsize=20)
# plt.ylabel('Latent dim 2', fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.subplot(132)
# plt.bar(np.arange(latent_dim), height=inv_lengthscale.detach().cpu().numpy().flatten())
# plt.title('Inverse Lengthscale of kernel', fontsize=18)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.subplot(133)
# plt.plot(loss_list, label='batch_size='+str(batch_size))
# plt.title('Neg. ELBO Loss', fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=14)
# # plt.show()
# plt.savefig(os.path.join('./results','mnist_DGP_'+str(model.num_layers)+'layers.png'),bbox_inches='tight')

# plot pairs
import pandas as pd
import seaborn as sns
dat2plot = pd.DataFrame(np.hstack((X[:,[l1,l2]],labels[:,None])),columns=['latdim_'+str(j) for j in range(2)]+['label'])
dat2plot['label']=dat2plot['label'].astype(int)
pairs = np.array([[0,6], [1,7], [2,3], [4,9], [5, 8]])
num_pairs = len(pairs)
num_pcls = 50
fig, axes = plt.subplots(1,num_pairs, figsize=(21,4))
for i, cls2plot in enumerate(pairs):
    plt.sca(axes[i])
    sns.kdeplot(data=dat2plot.iloc[np.where([lbl in cls2plot for lbl in labels])[0]], x='latdim_0', y='latdim_1', hue='label', palette=[colors[c] for c in cls2plot], fill=True, alpha=.5, legend=False)
    for c in cls2plot:
        idx = np.random.default_rng(seed).choice(np.where(labels==c)[0], size=num_pcls, replace=False)
        axes[i].scatter(X[idx, l1], X[idx, l2], c=[colors[c]], marker="$"+str(c)+"$")
    axes[i].set_title('2d latent of '+np.array2string(cls2plot,separator=','), fontsize=18)
    axes[i].set_xlabel('Latent dim 1', fontsize=16)
    axes[i].set_ylabel('Latent dim 2' if i==0 else '', fontsize=16)
    # axes[i].tick_params(axis='both', which='major', labelsize=12)
plt.subplots_adjust(wspace=0.15, hspace=0.15)
plt.savefig(os.path.join('./results','mnist_DGP_'+str(model.num_layers)+'layers_latentpairs.png'),bbox_inches='tight')