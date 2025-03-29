"Plot PCA"

import os
import random
import numpy as np
import matplotlib.pylab as plt

from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap, TSNE

# Setting manual seed for reproducibility
seed=2024
random.seed(seed)
np.random.seed(seed)

# create data
dataset = {0:'swissroll',1:'swisshole'}[1]
n_samples = 1000
sr_points, sr_color = make_swiss_roll(n_samples=n_samples, noise=0.05, random_state=0, hole='hole' in dataset)


latent_dim = 2
idx2plot = np.random.default_rng(seed).choice(n_samples, size=500, replace=False)
colors = sr_color[idx2plot]

# # plot Isomap
# isomap = Isomap(n_components=latent_dim)
# X = isomap.fit_transform(sr_points)[idx2plot]
#
# fig = plt.figure(figsize=(7, 6))
# plt.scatter(X[:, 0], X[:,1], c=colors, alpha=0.8)
# plt.title('Isomap', fontsize=25)
# # plt.axis('square')
# plt.xlabel('Latent dim 1', fontsize=20)
# plt.ylabel('Latent dim 2', fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=15)
# plt.savefig(os.path.join('./results_'+dataset,dataset+'_isomap.png'),bbox_inches='tight')

# plot t-SNE
tsne = TSNE(n_components=latent_dim, random_state=seed)
X = tsne.fit_transform(sr_points)[idx2plot]

fig = plt.figure(figsize=(7, 6))
plt.scatter(X[:, 0], X[:,1], c=colors, alpha=0.8)
plt.title('t-SNE', fontsize=25)
# plt.axis('square')
plt.xlabel('Latent dim 1', fontsize=20)
plt.ylabel('Latent dim 2', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.savefig(os.path.join('./results_'+dataset,dataset+'_tsne.png'),bbox_inches='tight')