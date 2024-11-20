"Plot PCA"

import os
import random
import numpy as np
import matplotlib.pylab as plt

import torch

# Setting manual seed for reproducibility
seed=2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# create data
from load_data import *
train, valid, test = load_weights()
Y = torch.tensor(np.vstack((train, valid)))
n_samples = Y.shape[0]

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection="3d")
# fig.add_axes(ax)
# ax.scatter(
#     Y[:, 0], Y[:, 1], Y[:, 2], alpha=0.8
# )
# ax.view_init(azim=-66, elev=12)
# os.makedirs('./results', exist_ok=True)
# plt.savefig(os.path.join('./results','robot.png'),bbox_inches='tight')

def _pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.matmul(Y, V[:,:latent_dim]), S

latent_dim = 2

idx2plot = np.random.default_rng(seed).choice(n_samples, size=500, replace=False)
X, S = _pca(Y, latent_dim)
X = X[idx2plot]

# fig = plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], X[:,1], alpha=0.8)
# plt.title('PCA', fontsize=20)
# plt.axis('square')
# plt.xlabel('Principal dim 1', fontsize=18)
# plt.ylabel('Principal dim 2', fontsize=18)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.savefig(os.path.join('./results','robot_pca.png'),bbox_inches='tight')

latent_dim = 10
_, S = _pca(Y, latent_dim)

fig = plt.figure(figsize=(7, 6))
plt.bar(np.arange(latent_dim), height=S)
plt.title('Dominant Eigenvalues of PCA', fontsize=20)
plt.ylabel(' ', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join('./results','robot_latdim_pca.png'),bbox_inches='tight')