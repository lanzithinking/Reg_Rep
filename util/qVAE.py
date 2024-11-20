#!/usr/bin/env python
"""
Variational Auto-Encoder (VAE)
by Diederik P Kingma and Max Welling, Auto-Encoding Variational Bayes, ICLR (2014)
----------------------------------------------------------------------------------
implemented with Q-Exponential distribution
by Shiwei Lan @ASU 2024
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2024, Regularization of Representation project"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com;"

import numpy as np
import torch
import torch.nn as nn
# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar  = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.training = True
        
    def forward(self, x):
        x = self.LeakyReLU(self.fc1(x))
        x = self.LeakyReLU(self.fc2(x))
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)                     # encoder produces mean and log of variance 
                                                    # (i.e., parateters of simple tractable normal distribution "q"
        return mu, log_var

# define decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, z):
        z = self.LeakyReLU(self.fc1(z))
        z = self.LeakyReLU(self.fc2(z))
        # x_hat = torch.sigmoid(self.output(z))
        x_hat = self.output(z)
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
        quad = torch.sum((x - x_hat)**2, dim=-1)
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
