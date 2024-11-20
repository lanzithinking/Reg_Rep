"Variatioanl AutoEncoder Model"

import os
import random
import numpy as np
import matplotlib.pylab as plt

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

# create data
from load_data import *
train, valid, test = load_weights()
Y = torch.tensor(np.vstack((train, valid)))
Y -= Y.mean(0, keepdim=True); Y /= Y.std(0, keepdim=True)
n_samples = Y.shape[0]
train_dataset = TensorDataset(Y, torch.arange(n_samples))
batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# define model
data_dim = Y.shape[1]
hidden_dim = 30
latent_dim = 10

# Model
model = VAE(data_dim, hidden_dim, latent_dim, power=POWER)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loss_list = []
if os.path.exists(os.path.join('./results', 'robot_vae_q'+str(q)+'_checkpoint.dat')):
    state_dict = torch.load(os.path.join('./results','robot_vae_q'+str(q)+'_checkpoint.dat'), map_location=device)['model']
else:
    # set device
    model = model.to(device)
    
    # Training loop - optimises the objective wrt variational params using the optimizer provided.
    model.train()
    
    os.makedirs('./results', exist_ok=True)
    # loss_list = []
    num_epochs = 10000
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
        # iterator.set_description('Loss: ' + str(float(np.round(loss_i,2))) + ", iter no: " + str(epoch))
        print('Epoch {}/{}: Loss: {}'.format(epoch, num_epochs, loss_i ))
    # save the model
    state_dict = optim_model#.state_dict()
    torch.save({'model': state_dict}, os.path.join('./results','robot_vae_q'+str(q)+'_checkpoint.dat'))

# load the best model
model.load_state_dict(state_dict)
model.eval()

# plot results
idx2plot = np.random.default_rng(seed).choice(n_samples, size=500, replace=False)

# _,_,log_vars = model(Y)
X, log_vars = model.encoder(Y.to(device))
vars = log_vars.exp().median(0)[0]
values, indices = torch.topk(vars, k=2)
l1, l2 = indices.detach().cpu().numpy().flatten()[:2]

# X = model._latent(Y)
X = X.detach().cpu().numpy()[idx2plot]

# # plot
# plt.figure(figsize=(20, 6))
# plt.subplot(131)
# plt.scatter(X[:, l1], X[:, l2], alpha=0.8)
# # plt.xlim([-1,1]); plt.ylim([-1,1])
# plt.title('2d latent subspace', fontsize=20)
# plt.xlabel('Latent dim 1', fontsize=20)
# plt.ylabel('Latent dim 2', fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.subplot(132)
# plt.bar(np.arange(latent_dim), height=vars.detach().cpu().numpy().flatten())
# plt.title('Variances of latent distribution', fontsize=18)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.subplot(133)
# plt.plot(loss_list, label='batch_size='+str(batch_size))
# plt.title('Neg. ELBO Loss', fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=14)
# # plt.show()
# os.makedirs('./results', exist_ok=True)
# plt.savefig(os.path.join('./results','robot_vae_q'+str(q)+'.png'),bbox_inches='tight')
#
# fig = plt.figure(figsize=(7, 6))
# plt.scatter(X[:, l1], X[:, l2], alpha=0.8)
# plt.title('VAE', fontsize=20)
# # plt.axis('square')
# plt.xlabel('Latent dim 1', fontsize=18)
# plt.ylabel('Latent dim 2', fontsize=18)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.savefig(os.path.join('./results','robot_latent_vae_q'+str(q)+'.png'),bbox_inches='tight')
#
# fig = plt.figure(figsize=(7, 6))
# plt.bar(np.arange(latent_dim), height=vars.detach().cpu().numpy().flatten())
# plt.title('Variances of latent distribution', fontsize=20)
# plt.ylabel(' ', fontsize=18)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.savefig(os.path.join('./results','robot_latdim_vae_q'+str(q)+'.png'),bbox_inches='tight')

# plot latent space
fig = plt.figure(figsize=(7,6))
from sklearn.manifold import TSNE
model_tsne = TSNE(n_components=2, random_state=seed)
Z_ = model_tsne.fit_transform(X)
plt.scatter(Z_[:, 0], Z_[:, 1])
plt.title('VAE', fontsize=20)
plt.xlabel('Latent dim 1', fontsize=20)
plt.ylabel('Latent dim 2', fontsize=20)
plt.savefig(os.path.join('./results','robot_vae_q'+str(q)+'_latent.png'),bbox_inches='tight')