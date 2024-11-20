#!/usr/bin/env python
"""
Variational Auto-Encoder (VAE)
by Diederik P Kingma and Max Welling, Auto-Encoding Variational Bayes, ICLR (2014)
----------------------------------------------------------------------------------
implemented with Q-Exponential distribution and CNN layers
by Shiwei Lan @ASU 2024
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2024, Regularization of Representation project"
__license__ = "GPL"
__version__ = "0.2"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com;"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define encoder
class Encoder(nn.Module):
    def __init__(self, img_sz, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * np.prod([i//4 for i in img_sz]), hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.training = True
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)                     # encoder produces mean and log of variance 
                                                    # (i.e., parateters of simple tractable normal distribution "q"
        return mu, log_var

# define decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, img_sz):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64 * np.prod([i//4 for i in img_sz]))
        self.reshape = lambda x: x.view(-1, 64, *[i//4 for i in img_sz])
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = self.reshape(z)
        z = F.relu(self.deconv1(z))
        x_hat = torch.sigmoid(self.deconv2(z))
        # x_hat = self.deconv2(z)
        return x_hat

# define VAE
class VAE(nn.Module):
    def __init__(self, data_dim, hidden_dim, latent_dim, power=torch.tensor(2.0, device=device)):
        super(VAE, self).__init__()
        self.encoder = Encoder(data_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, data_dim)
        self.power = power
        
    def reparameterization(self, mean, std):
        eps = torch.randn_like(std).to(device)        # sampling epsilon        
        if self.power!=2: eps *= torch.norm(eps, dim=-1, keepdim=True)**(2./self.power-1)
        z = mean + std*eps                          # reparameterization trick
        return z
        
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterization(mu, torch.exp(0.5 * log_var)) # takes exponential function (log var -> std)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var
    
    def loss(self, x, x_hat, mu, log_var, reduction='mean'):
        quad = torch.sum((x - x_hat)**2, dim=tuple(range(1,x.ndim)))
        recn_loss = .5*quad**(self.power/2.)
        if self.power!=2: recn_loss -= (self.power/2.-1)*quad.log()
        trace_quad = torch.sum(log_var.exp() + mu**2, dim=-1)
        kl_loss = 0.5*( trace_quad**(self.power/2.) -1 -log_var.sum(dim=-1))
        if self.power!=2: kl_loss += 0.5*( -(self.power/2.-1)*torch.log(trace_quad) + 0)
        total_loss = recn_loss + kl_loss
        return getattr(torch, reduction)(total_loss)
    
    def _latent(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterization(mu, torch.exp(0.5 * log_var))
        return z
