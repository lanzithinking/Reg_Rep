"Variatioanl AutoEncoder Model"

import os
import random
import numpy as np
import matplotlib.pylab as plt
import urllib.request
import tarfile

import torch
from torch.utils.data import TensorDataset, DataLoader
import tqdm

import sys
sys.path.insert(0,'../')
from util.qVAE import *


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
POWER = torch.tensor(q, device=device)

# load data
if not (os.path.exists('DataTrn.txt') and os.path.exists('DataTrnLbls.txt')):
    # download data
    url = "http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/resources/3PhData.tar.gz"
    urllib.request.urlretrieve(url, '3PhData.tar.gz')
    with tarfile.open('3PhData.tar.gz', 'r') as f:
        f.extract('DataTrn.txt')
        f.extract('DataTrnLbls.txt')

Y = torch.Tensor(np.loadtxt(fname='DataTrn.txt'))
Y -= Y.mean(0, keepdim=True); Y /= Y.std(0, keepdim=True)
labels = torch.Tensor(np.loadtxt(fname='DataTrnLbls.txt'))
labels = (labels @ np.diag([1, 2, 3])).sum(axis=1)
train_dataset = TensorDataset(Y, labels)
batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# define model
data_dim = Y.shape[1]
hidden_dim = data_dim
latent_dim = data_dim

# Model
model = VAE(data_dim, hidden_dim, latent_dim, power=POWER)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


loss_list = []
if os.path.exists(os.path.join('./results','oilflow_vae_q'+str(q)+'_checkpoint.dat')):
    state_dict = torch.load(os.path.join('./results','oilflow_vae_q'+str(q)+'_checkpoint.dat'), map_location=device)['model']
else:
    # set device
    model = model.to(device)
    
    # Training loop - optimises the objective wrt variational params using the optimizer provided.
    model.train()
    
    os.makedirs('./results', exist_ok=True)
    num_epochs = 1000
    iterator = tqdm.tqdm(range(num_epochs), desc="Epoch")
    for epoch in iterator:
        minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
        loss_i = 0
        for x, _ in minibatch_iter:
            if torch.cuda.is_available():
                x = x.cuda()
            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            loss = model.loss(x, x_hat, mean, log_var)
            loss.backward()
            optimizer.step()
            loss_i += loss.item()
            # minibatch_iter.set_postfix(loss=loss.item())
        loss_i /= len(minibatch_iter)
        # record the loss and the best model
        loss_list.append(loss_i)
        if epoch==0:
            min_loss = loss_list[-1]
            optim_model = model.state_dict()
        else:
            if loss_list[-1] < min_loss:
                min_loss = loss_list[-1]
                optim_model = model.state_dict()
        print('Epoch {}/{}: Loss: {}'.format(epoch, num_epochs, loss_i ))
    # save the model
    state_dict = optim_model#.state_dict()
    torch.save({'model': state_dict}, os.path.join('./results','oilflow_vae_q'+str(q)+'_checkpoint.dat'))

# load the best model
model.load_state_dict(state_dict)
model.eval()

# plot results
# _,_,log_vars = model(Y)
X, log_vars = model.encoder(Y.to(device))
vars = log_vars.exp().median(0)[0]
values, indices = torch.topk(vars, k=2)
l1, l2 = indices.detach().cpu().numpy().flatten()[:2]

# X = model._latent(Y)
X = X.detach().cpu().numpy()
std = torch.exp(0.5*log_vars).detach().cpu().numpy()
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
# plt.bar(np.arange(latent_dim), height=vars.detach().cpu().numpy().flatten())
# plt.title('Variances of latent distribution', fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.subplot(133)
# plt.plot(loss_list, label='batch_size=100')
# plt.title('Neg. ELBO Loss', fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=14)
# # plt.show()
# plt.savefig(os.path.join('./results','./oilflow_vae_q'+str(q)+'.png'),bbox_inches='tight')

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
plt.title('VAE', fontsize=25)
plt.xlabel('Latent dim 1', fontsize=20)
plt.ylabel('Latent dim 2', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.savefig(os.path.join('./results','./oilflow_latent_vae_q'+str(q)+'.png'),bbox_inches='tight')

plt.figure(figsize=(7, 6))
plt.bar(np.arange(latent_dim), height=vars.detach().cpu().numpy().flatten())
plt.title('Variances of latent distribution', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.savefig(os.path.join('./results','./oilflow_latdim_vae_q'+str(q)+'.png'),bbox_inches='tight')