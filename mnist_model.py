import torch
import torch.nn as nn

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

    def encode(self, x):
        z = self.encoder(x)
        return z[:, :self.z_dim], z[:, self.z_dim:]

    def sample(self, x):
        post_dist = self.forward(x)
        return post_dist.rsample()

    def forward(self, x):
        mu, logvar = self.encode(x)
        post_dist = torch.distributions.Normal(mu, torch.exp(0.5*logvar))
        return post_dist
    
class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        
        self.z_dim = z_dim
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
        
    def decode(self, z):
        logits = self.decoder(z)
        return logits

    def sample(self, z):
        gen_dist = self.forward(z)
        return gen_dist.sample()

    def forward(self, z):
        logits = self.decode(z)
        gen_dist = torch.distributions.Bernoulli(logits=logits)
        return gen_dist, logits
    
class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        
        self.z_dim = z_dim
        self.posterior = Posterior(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x):
        post_dist = self.posterior(x)
        z = post_dist.rsample()
        gen_dist, logits = self.decoder(z)
        return post_dist, gen_dist, z, torch.nn.functional.sigmoid(logits)

    def sample(self, batch_size=10):
        z = torch.randn(batch_size, self.z_dim).cuda()
        gen_dist, logits = self.decoder(z)
        return torch.nn.functional.sigmoid(logits)

    def elbo(self, x):
        post_dist, gen_dist, z, _ = self.forward(x)
        prior = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
        recon_term = gen_dist.log_prob(torch.bernoulli(x)).sum()/x.size(0)
        kl_term = torch.distributions.kl_divergence(post_dist, prior).sum()/x.size(0)
        elbo = recon_term - kl_term
        return elbo, recon_term, kl_term