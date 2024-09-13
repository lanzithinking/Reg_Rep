"Plot PCA"

import os
import random
import numpy as np
import matplotlib.pylab as plt

from sklearn.datasets import make_swiss_roll
import torch

# Setting manual seed for reproducibility
seed=2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# create data
dataset = {0:'swissroll',1:'swisshole'}[1]
n_samples = 1000
sr_points, sr_color = make_swiss_roll(n_samples=n_samples, noise=0.05, random_state=0, hole='hole' in dataset)
Y, t = torch.tensor(sr_points), torch.tensor(sr_color)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
fig.add_axes(ax)
ax.scatter(
    sr_points[:, 0], sr_points[:, 1], sr_points[:, 2], c=sr_color, alpha=0.8
)
# ax.set_title("Swiss Roll", fontsize=18)# in Ambient Space")
ax.view_init(azim=-66, elev=12)
os.makedirs('./results', exist_ok=True)
plt.savefig(os.path.join('./results',dataset+'.png'),bbox_inches='tight')

def _pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.matmul(Y, V[:,:latent_dim])

latent_dim = 2

plt.figure(figsize=(20, 6))

idx2plot = np.random.default_rng(seed).choice(n_samples, size=500, replace=False)
X = _pca(Y, latent_dim)[idx2plot]
colors = t[idx2plot]

fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:,1], c=colors, alpha=0.8)
plt.title('PCA', fontsize=20)
plt.axis('square')
plt.xlabel('Principal dim 1', fontsize=18)
plt.ylabel('Principal dim 2', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join('./results',dataset+'_pca.png'),bbox_inches='tight')