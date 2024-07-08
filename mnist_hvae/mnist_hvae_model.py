import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence as KL

class Posterior(nn.Module):
    def __init__(self, z_dim):
        super(Posterior, self).__init__()
        
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64*4*4, z_dim*2)
        )
        self.z_layers = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, z_dim*2)
        )

    def forward(self, x, detach_z=False):
        z_1_params = self.encoder(x)
        z_1_dist = torch.distributions.Normal(z_1_params[: , :self.z_dim], torch.exp(0.5*z_1_params[:, self.z_dim:]))
        z_1 = z_1_dist.rsample()
        if detach_z:
            z_1 = z_1.detach()
        z_2_params = self.z_layers(z_1)
        z_2_dist = torch.distributions.Normal(z_2_params[: , :self.z_dim], torch.exp(0.5*z_2_params[:, self.z_dim:]))
        z_2 = z_2_dist.rsample()
        if detach_z:
            z_2 = z_2.detach()
        return z_1_dist, z_2_dist, z_1, z_2
 
    
class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        
        self.z_dim = z_dim
        self.z_layers = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, z_dim*2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 64*4*4),
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
        )
        
    def forward(self, batch_size=10):
        z_2_dist = torch.distributions.Normal(torch.zeros((batch_size, self.z_dim)).cuda(), torch.ones((batch_size, self.z_dim)).cuda())
        z_2 = z_2_dist.rsample()
        z_1_params = self.z_layers(z_2)
        z_1_dist = torch.distributions.Normal(z_1_params[: , :self.z_dim], torch.exp(0.5*z_1_params[:, self.z_dim:]))
        z_1 = z_1_dist.rsample()
        logits = self.decoder(z_1)
        x_dist = torch.distributions.Bernoulli(logits=logits)
        x_pred = torch.nn.functional.sigmoid(logits)
        return x_dist, z_1_dist, z_2_dist, x_pred, z_1, z_2

    
class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        
        self.z_dim = z_dim
        self.posterior = Posterior(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x):
        z_1_dist, z_2_dist, z_1, z_2 = self.posterior(x)
        elbo, recon_term, kl_term, kl_1, kl_2 = self.elbo(x, z_1, z_2, z_1_dist, z_2_dist)
        return elbo, recon_term, kl_term, kl_1, kl_2

    def elbo(self, x, z_1, z_2, z_1_dist, z_2_dist):
        p_z_2_dist = torch.distributions.Normal(torch.zeros_like(z_2), torch.ones_like(z_2))
        kl_2 = KL(z_2_dist, p_z_2_dist).sum()/x.size(0)
        p_z_1_params =  self.decoder.z_layers(z_2)
        p_z_1_dist = torch.distributions.Normal(p_z_1_params[: , :self.z_dim], torch.exp(0.5*p_z_1_params[:, self.z_dim:]))
        kl_1 = KL(z_1_dist, p_z_1_dist).sum()/x.size(0)
        kl_term = kl_1 + kl_2
        logits = self.decoder.decoder(z_1)
        x_dist = torch.distributions.Bernoulli(logits=logits)
        recon_term = x_dist.log_prob(torch.bernoulli(x)).sum()/x.size(0)
        elbo = recon_term - kl_term
        
        return elbo, recon_term, kl_term, kl_1, kl_2