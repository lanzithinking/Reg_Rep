"Variatioanl AutoEncoder Model"

import os
import random
import numpy as np
import matplotlib.pylab as plt

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tqdm

import sys
sys.path.insert(0,'../')

use_CNN = False
if use_CNN:
    from util.qVAE_CNN import *
else:
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
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])
train_dataset = datasets.MNIST('./data/MNIST', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data/MNIST', train=False, download=True, transform=transform)

labels = train_dataset.targets
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# define model
img_sz = train_dataset.data.shape[-2:]
data_dim = np.prod(img_sz)
hidden_dim = 200
latent_dim = 100

# Model
model = VAE(img_sz if use_CNN else data_dim, hidden_dim, latent_dim, power=POWER)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_list = []
if os.path.exists(os.path.join('./results','mnist_vae_q'+str(q)+'_checkpoint.dat')):
    state_dict = torch.load(os.path.join('./results','mnist_vae_q'+str(q)+'_checkpoint.dat'), map_location=device)['model']
else:
    # set device
    model = model.to(device)
    
    # Training loop - optimises the objective wrt variational params using the optimizer provided.
    model.train()
    
    os.makedirs('./results', exist_ok=True)
    # loss_list = []
    num_epochs = 1000
    iterator = tqdm.tqdm(range(num_epochs), desc="Epoch")
    for epoch in iterator:
        minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
        loss_i = 0
        for x, _ in minibatch_iter:
            if torch.cuda.is_available():
                x = x.cuda()
            if not use_CNN:
                x = x.view(-1, data_dim)
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
    torch.save({'model': state_dict}, os.path.join('./results','mnist_vae_q'+str(q)+'_checkpoint.dat'))

# load the best model
model.load_state_dict(state_dict)
model.eval()

# plot results
idx2plot = np.random.default_rng(seed).choice(train_dataset.data.shape[0], size=1000, replace=False)

# x_hat,_,log_vars = model(train_dataset.data[idx2plot].to(device, dtype=torch.float32).unsqueeze(1) if use_CNN else \
#                          train_dataset.data[idx2plot].to(device, dtype=torch.float32).view(-1, data_dim))
Z_,log_vars = model.encoder(train_dataset.data[idx2plot].to(device, dtype=torch.float32).unsqueeze(1)if use_CNN else \
                            train_dataset.data[idx2plot].to(device, dtype=torch.float32).view(-1, data_dim))
vars = log_vars.exp().median(0)[0]
values, indices = torch.topk(vars, k=2)
l1, l2 = indices.detach().cpu().numpy().flatten()[:2]

# Z_ = model._latent(train_dataset.data[idx2plot].to(device, dtype=torch.float32).unsqueeze(1))
Z_ = Z_.detach().cpu().numpy()
labels = labels.numpy()

# plot
import matplotlib.colors as mcolors
colors = list(mcolors.TABLEAU_COLORS.values())

# plt.figure(figsize=(20, 6))
# # idx2plot = model._get_batch_idx(500, seed)
cls2plot = np.unique(labels)
num_pcls = 100
idx2plot = []
for c in cls2plot:
    idx2plot.append(np.random.default_rng(seed).choice(np.where(labels==c)[0], size=num_pcls, replace=False))
idx2plot = np.concatenate(idx2plot)
# Z_ = Z[idx2plot]
labels_ = labels[idx2plot]
#
# plt.subplot(131)
# # std_ = torch.nn.functional.softplus(model.X.q_log_sigma).detach().numpy()[idx2plot]
# # Select index of the smallest lengthscales by examining model.covar_module.base_kernel.lengthscales
# for i, label in enumerate(np.unique(labels_)):
#     X_i = X_[labels_ == label]
#     # scale_i = std_[labels_ == label]
#     # plt.scatter(X_i[:, l1], X_i[:, l2], c=[colors[i]], label=label)
#     plt.scatter(X_i[:, l1], X_i[:, l2], c=[colors[i]], marker="$"+str(label)+"$")
#     # plt.errorbar(X_i[:, l1], X_i[:, l2], xerr=scale_i[:,l1], yerr=scale_i[:,l2], label=label,c=colors[i], fmt='none')
# # plt.xlim([-1,1]); plt.ylim([-1,1])
# plt.title('2d latent subspace', fontsize=20)
# plt.xlabel('Latent dim 1', fontsize=20)
# plt.ylabel('Latent dim 2', fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.subplot(132)
# plt.bar(np.arange(latent_dim), height=inv_lengthscale.detach().cpu().numpy().flatten())
# plt.title('Inverse Lengthscale of SE-ARD kernel', fontsize=18)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.subplot(133)
# plt.plot(loss_list, label='batch_size='+str(batch_size))
# plt.title('Neg. ELBO Loss', fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=14)
# # plt.show()
# plt.savefig(os.path.join('./results','mnist_QEP-LVM_q'+str(POWER.cpu().item())+'.png'),bbox_inches='tight')

# plot latent space
fig = plt.figure(figsize=(7,6))
from sklearn.manifold import TSNE
model_tsne = TSNE(n_components=2, random_state=seed)
Z__ = model_tsne.fit_transform(Z_)
for i, label in enumerate(np.unique(labels_)):
    Z_i = Z__[labels_ == label]
    plt.scatter(Z_i[:, 0], Z_i[:, 1], c=[colors[i]], marker="$"+str(label)+"$")#label=label)
# plt.title("Latent Variable per Class", fontsize=20)
plt.title('VAE', fontsize=25)
plt.xlabel('Latent dim 1', fontsize=20)
plt.ylabel('Latent dim 2', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.savefig(os.path.join('./results','mnist_vae_q'+str(q)+'_latent.png'),bbox_inches='tight')

# # plot latent pairs
# import pandas as pd
# import seaborn as sns
# dat2plot = pd.DataFrame(np.hstack((Z_[:,[l1,l2]],labels_[:,None])),columns=['latdim_'+str(j) for j in range(2)]+['label'])
# dat2plot['label']=dat2plot['label'].astype(int)
# pairs = np.array([[0,6], [1,7], [2,3], [4,9], [5, 8]])
# num_pairs = len(pairs)
# num_pcls = 50
# fig, axes = plt.subplots(1,num_pairs, figsize=(21,4))
# for i, cls2plot in enumerate(pairs):
#     plt.sca(axes[i])
#     sns.kdeplot(data=dat2plot.iloc[np.where([lbl in cls2plot for lbl in labels_])[0]], x='latdim_0', y='latdim_1', hue='label', palette=[colors[c] for c in cls2plot], fill=True, alpha=.5, legend=False)
#     for c in cls2plot:
#         idx = np.random.default_rng(seed).choice(np.where(labels_==c)[0], size=num_pcls, replace=False)
#         axes[i].scatter(Z_[idx, l1], Z_[idx, l2], c=[colors[c]], marker="$"+str(c)+"$")
#     axes[i].set_title('2d latent of '+np.array2string(cls2plot,separator=','), fontsize=18)
#     axes[i].set_xlabel('Latent dim 1', fontsize=16)
#     axes[i].set_ylabel('Latent dim 2' if i==0 else '', fontsize=16)
#     # axes[i].tick_params(axis='both', which='major', labelsize=12)
# plt.subplots_adjust(wspace=0.15, hspace=0.15)
# plt.savefig(os.path.join('./results','mnist_vae_q'+str(q)+'_latentpairs.png'),bbox_inches='tight')
#
# # plot samples
# with torch.no_grad():
#     for x, labels_ in test_loader:
#         if torch.cuda.is_available():
#             x = x.cuda()
#         if not use_CNN:
#             x = x.view(-1, data_dim)
#         x_hat, _, _ = model(x)
#         break
#
# fig, axes = plt.subplots(2,5, figsize=(20,8))
# for i,ax in enumerate(axes.flatten()):
#     plt.sca(ax)
#     idx_i = np.random.default_rng(seed).choice(np.where(labels_==i)[0], size=1)
#     sample_digit = x_hat[idx_i].squeeze().detach().cpu().numpy().reshape(*img_sz)
#     sample_digit = (sample_digit - sample_digit.min())/(sample_digit.max()-sample_digit.min())
#     plt.imshow(sample_digit, cmap='gray')#'Greys')
#     plt.axis('off')
# plt.subplots_adjust(wspace=0.1, hspace=0.1)
# plt.savefig(os.path.join('./results','mnist_vae_q'+str(q)+'_sampledigits.png'),bbox_inches='tight')