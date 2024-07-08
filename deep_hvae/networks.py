import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence as KL


class PositionalNorm(nn.LayerNorm):
    """PositionalNorm is a normalization layer used for 3D image inputs that
    normalizes exclusively across the channels dimension.
    https://arxiv.org/abs/1907.04312
    """

    def forward(self, x):
        # The input is of shape (B, C, H, W). Transpose the input so that the
        # channels are pushed to the last dimension and then run the standard
        # LayerNorm layer.
        #x = x.permute(0, 2, 3, 1).contiguous()
        #out = super().forward(x)
        #out = out.permute(0, 3, 1, 2).contiguous()
        return x


class ResBlock(nn.Module):
    """Residual block following the "bottleneck" architecture as described in
    https://arxiv.org/abs/1512.03385. See Figure 5.
    The residual blocks are defined following the "pre-activation" technique
    as described in https://arxiv.org/abs/1603.05027.

    The residual block applies a number of convolutions represented as F(x) and
    then skip connects the input to produce the output F(x)+x. The residual
    block can also be used to upscale or downscale the input by doubling or
    halving the spatial dimensions, respectively. Scaling is performed by the
    "bottleneck" layer of the block. If the residual block changes the number of
    channels, or the spatial dimensions are up- or down-scaled, then the input
    is also transformed into the desired shape for the addition operation.
    """

    def __init__(self, in_chan, out_chan, scale="same", return_std=False):
        """Init a Residual block.

        Args:
            in_chan: int
                Number of channels of the input tensor.
            out_chan: int
                Number of channels of the output tensor.
            scale: string, optional
                One of ["same", "upscale", "downscale"].
                Upscale or downscale by half the spatial dimensions of the
                input tensor. Default is "same", i.e., no scaling.
        """
        super().__init__()
        assert scale in ["same", "upscale", "downscale"]
        self.return_std = return_std
        self.softplus = nn.Softplus()
        if scale == "same":
            bottleneck = nn.Conv2d(in_chan//2, in_chan//2, kernel_size=3, padding="same")
            stride = 1
        elif scale == "downscale":
            bottleneck = nn.Conv2d(in_chan//2, in_chan//2, kernel_size=3, stride=2, padding=1)
            stride = 2
        elif scale == "upscale":
            bottleneck = nn.ConvTranspose2d(in_chan//2, in_chan//2, kernel_size=4, stride=2, padding=1)
            stride = 1

        # The residual block employs the bottleneck architecture as described
        # in Sec 4. under the paragraph "Deeper Bottleneck Architectures" of the
        # original paper introducing the ResNet architecture.
        # The block uses a stack of three layers: `1x1`, `3x3` (`4x4`), `1x1`
        # convolutions. The first `1x1` reduces (in half) the number of channels
        # before the expensive `3x3` (`4x4`) convolution. The second `1x1`
        # up-scales the channels to the requested output channel size.
        self.block = nn.Sequential(
            # 1x1 convolution
            PositionalNorm(in_chan),
            nn.ReLU(),
            nn.Conv2d(in_chan, in_chan//2, kernel_size=1),

            # 3x3 convolution if same or downscale, 4x4 transposed convolution if upscale
            PositionalNorm(in_chan//2),
            nn.ReLU(),
            bottleneck,

            # 1x1 convolution
            PositionalNorm(in_chan//2),
            nn.ReLU(),
            nn.Conv2d(in_chan//2, out_chan, kernel_size=1),
        )

        # If channels or spatial dimensions are modified then transform the
        # input into the desired shape, otherwise use a simple identity layer.
        self.id = nn.Identity()
        if (in_chan != out_chan or scale == "downscale") and return_std == False:
            # We will downscale by applying a strided `1x1` convolution.
            self.id = nn.Sequential(
                PositionalNorm(in_chan),
                nn.ReLU(),
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride),
            )
        elif (in_chan != out_chan or scale == "downscale") and return_std == True:
            self.id = nn.Sequential(
                PositionalNorm(in_chan),
                nn.ReLU(),
                nn.Conv2d(in_chan, out_chan//2, kernel_size=1, stride=stride),
            )
        if scale == "upscale":
            # We will upscale by applying a nearest-neighbor upsample.
            # Channels are again modified using a `1x1` convolution.
            self.id = nn.Sequential(
                PositionalNorm(in_chan),
                nn.ReLU(),
                nn.Conv2d(in_chan, out_chan, kernel_size=1),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

    def forward(self, x):
        if self.return_std:
            z_params = self.block(x)
            z_mu, z_std = torch.chunk(z_params, 2, dim=1)
            z_mu = z_mu + self.id(x)
            z_std = torch.exp(0.5*z_std) #self.softplus(z_std)
            return torch.cat([z_mu, z_std], dim=1)
        else:
            return self.block(x) + self.id(x)


class Posterior(nn.Module):
    """Encoder network used for encoding the input space into a latent space.
    The encoder maps a vector(tensor) from the input space into a distribution
    over latent space.
    """

    def __init__(self, in_chan=3, latent_channels=8, depth=1):
        """Init an Encoder module.

        Args:
            in_chan: int
                Number of input channels of the images.
            latent_channels: int
                Channels of the latent space.
            depth: int
                Number of latent blocks.
        """
        super().__init__()

        # The encoder architecture follows the design of ResNet stacking several
        # residual blocks into groups, operating on different scales of the image.
        # The first residual block from each group is responsible for downsizing
        # the image and increasing the channels.
        self.latent_channels = latent_channels
        self.depth = depth
        
        self.encoder = nn.Sequential(
            # Stem.
            nn.Conv2d(in_chan, 32, kernel_size=3, padding="same"),  # 32x32

            # Body.
            ResBlock(in_chan=32, out_chan=64, scale="downscale"),   # 16x16
            ResBlock(in_chan=64, out_chan=64),
            ResBlock(in_chan=64, out_chan=64),
            ResBlock(in_chan=64, out_chan=64),

            ResBlock(in_chan=64, out_chan=128, scale="downscale"),  # 8x8
            ResBlock(in_chan=128, out_chan=128),
            ResBlock(in_chan=128, out_chan=128),
        )
        
        self.latent_blocks = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                input_channels = 128
            else:
                input_channels = latent_channels
            z_block = nn.Sequential(
                ResBlock(in_chan=input_channels, out_chan=128),
                ResBlock(in_chan=128, out_chan=128),
                ResBlock(in_chan=128, out_chan=128),
                ResBlock(in_chan=128, out_chan=2*latent_channels, return_std=True)
            )
            self.latent_blocks.append(z_block)

    def forward(self, x, detach_z=False):
        """Forward the input through the encoder and return the parameters `mu`
        and `std` of the Normal distribution.
        """
        z = self.encoder(x)
        z_list = []
        z_dist_list = []
        log_prob_list = []
        for i in range(self.depth):
            z_params = self.latent_blocks[i](z)
            z_mu, z_std = torch.chunk(z_params, chunks=2, dim=1)
            z_dist = torch.distributions.Normal(z_mu, z_std)
            if detach_z:
                z = z_dist.sample()
            else:
                z = z_dist.rsample()
            log_prob = z_dist.log_prob(z).sum(dim=(1,2,3))
            log_prob_list.append(log_prob)
            z_list.append(z)
            z_dist_list.append(z_dist)
        # Return reversed list (From z_{depth-1} to z_0)
        return list(reversed(z_list)), list(reversed(z_dist_list)), list(reversed(log_prob_list))
    
    def traj_log_prob(self, x, z_list): # z_list: list of z_{depth-1} to z_0
        z_list = list(reversed(z_list))
        log_prob_list = []
        z_dist_list = []
        z = self.encoder(x)
        for i in range(self.depth):
            z_params = self.latent_blocks[i](z)
            z_mu, z_std = torch.chunk(z_params, chunks=2, dim=1)
            z_dist = torch.distributions.Normal(z_mu, z_std)
            z = z_list[i]
            z_dist_list.append(z_dist)
            log_prob = z_dist.log_prob(z).sum(dim=(1,2,3))
            log_prob_list.append(log_prob)
        return list(reversed(z_dist_list)), list(reversed(log_prob_list))


class Generator(nn.Module):
    """Decoder network used for decoding the latent space back into the input
    space. The decoder maps a vector(tensor) from the latent space into a
    distribution over the input space.
    """

    def __init__(self, out_chan=3, latent_channels=8, depth=1, x_dist='gaussian'): # x_dist: 'gaussian' or 'bernoulli'
        """Init an Decoder module.

        Args:
            out_chan: int
                Number of output channels of the images.
            latent_dim: int
                Dimensionality of the latent space.
        """
        super().__init__()
        self.softplus = nn.Softplus()
        self.depth = depth
        self.x_dist = x_dist
        self.latent_channels = latent_channels
        self.latent_blocks = nn.ModuleList()
        # depth - 1 latent blocks because first latent is from Gaussian prior
        for i in range(depth-1):
            z_block = nn.Sequential(
                ResBlock(in_chan=latent_channels, out_chan=128),
                ResBlock(in_chan=128, out_chan=128),
                ResBlock(in_chan=128, out_chan=128),
                ResBlock(in_chan=128, out_chan=2*latent_channels, return_std=True)
            )
            self.latent_blocks.append(z_block)
        
        if x_dist == 'gaussian':
            out_chan = 2*out_chan
        self.decoder = nn.Sequential(
            # Body.
            ResBlock(in_chan=latent_channels, out_chan=128),   # 8x8
            ResBlock(in_chan=128, out_chan=128),
            ResBlock(in_chan=128, out_chan=128),

            ResBlock(in_chan=128, out_chan=64, scale="upscale"),    # 16x16
            ResBlock(in_chan=64, out_chan=64),
            ResBlock(in_chan=64, out_chan=64),
            ResBlock(in_chan=64, out_chan=64),

            ResBlock(in_chan=64, out_chan=32, scale="upscale"),     # 32x32
            ResBlock(in_chan=32, out_chan=32),
            ResBlock(in_chan=32, out_chan=32),
            ResBlock(in_chan=32, out_chan=32),

            # Inverse stem.
            PositionalNorm(32),
            nn.ReLU(),
            nn.Conv2d(32, out_chan, kernel_size=3, padding="same"),
        )

    def forward(self, batch_size=10):
        """Forward the input through the encoder and return the parameters `mu`
        and `std` of the Normal distribution.
        """
        z_list = []
        z_dist_list = []
        log_prob_list = []
        
        z_dist = torch.distributions.Normal(torch.zeros((batch_size, self.latent_channels, 8, 8)).cuda(), torch.ones((batch_size, self.latent_channels, 8 , 8)).cuda())
        z = z_dist.sample()
        log_prob = z_dist.log_prob(z).sum(dim=(1,2,3))
        log_prob_list.append(log_prob)
        z_list.append(z)
        z_dist_list.append(z_dist)
        
        for i in range(self.depth-1):
            z_params = self.latent_blocks[i](z)
            z_mu, z_std = torch.chunk(z_params, chunks=2, dim=1)
            z_dist = torch.distributions.Normal(z_mu, z_std)
            z = z_dist.sample()
            log_prob = z_dist.log_prob(z).sum(dim=(1,2,3))
            log_prob_list.append(log_prob)
            z_list.append(z)
            z_dist_list.append(z_dist)
            
        x_params = self.decoder(z)
        if self.x_dist == 'gaussian':
            x_mu, x_std = torch.chunk(x_params, chunks=2, dim=1)
            x_dist = torch.distributions.Normal(x_mu, torch.exp(0.5*x_std)) #self.softplus(x_std))
        elif self.x_dist == 'bernoulli':
            x_dist = torch.distributions.Bernoulli(logits=x_params)
        x_sample = x_dist.sample()
        log_prob = x_dist.log_prob(x_sample).sum(dim=(1,2,3))
        log_prob_list.append(log_prob)
        z_list.append(x_sample)
        z_dist_list.append(x_dist)
        # Return list from z_{depth-1} to z_0, x
        return z_list, z_dist_list, log_prob_list
    
    def traj_log_prob(self, x, z_list): # z_list: list of z_{depth-1} to z_0
        z_dist_list = []
        log_prob_list = []
        
        z_dist = torch.distributions.Normal(torch.zeros((x.shape[0], self.latent_channels, 8, 8)).cuda(), torch.ones((x.shape[0], self.latent_channels, 8 , 8)).cuda())
        log_prob = z_dist.log_prob(z_list[0]).sum(dim=(1,2,3))
        z_dist_list.append(z_dist)
        log_prob_list.append(log_prob)
        
        for i in range(self.depth-1):
            z_params = self.latent_blocks[i](z_list[i])
            z_mu, z_std = torch.chunk(z_params, chunks=2, dim=1)
            z_dist = torch.distributions.Normal(z_mu, z_std)
            log_prob = z_dist.log_prob(z_list[i+1]).sum(dim=(1,2,3))
            z_dist_list.append(z_dist)
            log_prob_list.append(log_prob)
            
        if self.x_dist == 'gaussian':
            x_mu, x_std = torch.chunk(self.decoder(z_list[-1]), chunks=2, dim=1)
            x_dist = torch.distributions.Normal(x_mu, torch.exp(0.5*x_std)) #self.softplus(x_std))
            log_prob = x_dist.log_prob(x).sum(dim=(1,2,3))
        elif self.x_dist == 'bernoulli':
            x_dist = torch.distributions.Bernoulli(logits=self.decoder(z_list[-1]))
            log_prob = x_dist.log_prob(torch.bernoulli(x)).sum(dim=(1,2,3))
        log_prob_list.append(log_prob)
        z_dist_list.append(x_dist)
        return z_dist_list, log_prob_list
        
    
class VAE(nn.Module):
    """Variational Autoencoder (VAE) model that combines the encoder and decoder
    networks into a single model.
    """

    def __init__(self, in_chan=3, out_chan=3, latent_channels=8, depth=1, x_dist='gaussian'):
        """Init a VAE model.

        Args:
            in_chan: int
                Number of input channels of the images.
            out_chan: int
                Number of output channels of the images.
            latent_dim: int
                Dimensionality of the latent space.
        """
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.latent_channels = latent_channels
        self.depth = depth
        self.x_dist = x_dist
        self.posterior = Posterior(in_chan=in_chan, latent_channels=latent_channels, depth=depth)
        self.generator = Generator(out_chan=out_chan, latent_channels=latent_channels, depth=depth, x_dist=x_dist)

    def forward(self, x, detach_z=False, no_grad_posterior=False):
        """
        Sample trajectory from posterior, and return dists and log probs of trajectory under Posterior and Generator
        """
        if no_grad_posterior:
            with torch.no_grad():
                z_list, q_dist_list, q_log_prob_list = self.posterior(x, detach_z=detach_z)
        else:
            z_list, q_dist_list, q_log_prob_list = self.posterior(x, detach_z=detach_z)
        p_dist_list, p_log_prob_list = self.generator.traj_log_prob(x, z_list)
        return q_dist_list, q_log_prob_list, p_dist_list, p_log_prob_list
    
    def elbo(self, x, detach_z=False, no_grad_posterior=False):
        """
        Compute ELBO
        """
        q_dist_list, _, p_dist_list, p_log_prob_list = self.forward(x, detach_z=detach_z, no_grad_posterior=no_grad_posterior)
        kl_list = [KL(q_dist_list[i], p_dist_list[i]).sum()/x.size(0) for i in range(self.depth)]
        kl_term = sum(kl_list)
        recon_term = p_log_prob_list[-1].mean()
        elbo = recon_term - kl_term
        return elbo, recon_term, kl_term, kl_list