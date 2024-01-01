import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GVAE(nn.Module):
    def __init__(self, in_features, hidden_dim, latent_dim):
        super(GVAE, self).__init__()
        self.gc1 = GCNConv(in_features, hidden_dim)
        self.gc2_mean = GCNConv(hidden_dim, latent_dim)
        self.gc2_logstd = GCNConv(hidden_dim, latent_dim)

    def encode(self, x, edge_index):
        hidden = F.relu(self.gc1(x, edge_index))
        return self.gc2_mean(hidden, edge_index), self.gc2_logstd(hidden, edge_index)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Implement the decoder depending on your graph reconstruction needs
        # Typically it involves reconstructing the adjacency matrix
        pass

    def forward(self, x, edge_index):
        mu, logstd = self.encode(x, edge_index)
        z = self.reparameterize(mu, logstd)
        return self.decode(z), mu, logstd


def loss_function(recon_x, x, mu, logvar):
    reconstruction_loss = F.mse_loss(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence
