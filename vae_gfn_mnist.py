import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb
from mnist_model import VAE
import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100000)
parser.add_argument('--log_image_freq', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--vargrad_batch_size', type=int, default=64)
parser.add_argument('--z_dim', type=int, default=32)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
parser.add_argument("--wandb-project-name", type=str, default="advantage-diffusion",
    help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default='swish',
    help="the entity (team) of wandb's project")
parser.add_argument("--run-name", type=str, default="vae-mnist")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.track:
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        config=vars(args),
        name=args.run_name
    )

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((32, 32))])
                                #transforms.Normalize((0.5,), (1.0,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

vae = VAE(z_dim=args.z_dim).to(device)
q_opt = optim.Adam(vae.posterior.parameters(), lr=args.lr)
p_opt = optim.Adam(vae.decoder.parameters(), lr=args.lr)

for epoch in range(args.n_epochs):
    elbo_epoch = 0.0
    recon_epoch = 0.0
    kl_epoch = 0.0
    elbo_epoch_test = 0.0
    recon_epoch_test = 0.0
    kl_epoch_test = 0.0
    tb_epoch = 0.0
    
    # Posterior TB training
    for i, (x, _) in enumerate(dataloader):
        x = x.to(device)
        x = torch.clamp(x, 0.0, 1.0)
        x = torch.repeat_interleave(x, args.vargrad_batch_size, dim=0)
        
        post_dist = vae.posterior(x)
        z = post_dist.sample()
        log_pb = post_dist.log_prob(z).sum(1)
        prior_dist = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
        log_pf = prior_dist.log_prob(z).sum(1)
        with torch.no_grad():
            gen_dist, logits = vae.decoder(z)
            log_pf += gen_dist.log_prob(torch.bernoulli(x)).sum(dim=(1, 2, 3))
        # log q(z|x) = log p(z) + log p(x|z) - log p(x) => log p(x) = -log q(z_1|x) + log p(z_1) + log p(x|z_1) = -log q(z_2|x) + log p(z_2) + log p(x|z_2)
        log_Z = (log_pf - log_pb).view(args.batch_size, args.vargrad_batch_size).mean(1).repeat_interleave(args.vargrad_batch_size, dim=0).detach()
        tb_loss = ((log_pb + log_Z - log_pf)**2).mean()
        q_opt.zero_grad()
        tb_loss.backward()
        q_opt.step()
        tb_epoch += tb_loss.item()
        
    # Decoder training
    for i, (x, _) in enumerate(dataloader):
        x = x.to(device)
        x = torch.clamp(x, 0.0, 1.0)
        elbo, recon_term, kl_term = vae.elbo(x)
        loss = -elbo
        p_opt.zero_grad()
        loss.backward()
        p_opt.step()
        elbo_epoch += elbo.item()
        recon_epoch += recon_term.item()
        kl_epoch += kl_term.item()

    with torch.no_grad():
        for j, (x, _) in enumerate(testloader):
            x = x.to(device)
            x = torch.clamp(x, 0.0, 1.0)
            elbo, recon_term, kl_term = vae.elbo(x)
            elbo_epoch_test += elbo.item()
            recon_epoch_test += recon_term.item()
            kl_epoch_test += kl_term.item()
    elbo_epoch /= i
    recon_epoch /= i
    kl_epoch /= i
    tb_epoch /= i
    elbo_epoch_test /= j
    recon_epoch_test /= j
    kl_epoch_test /= j
    print(f"Epoch: {epoch}, ELBO: {elbo_epoch}, recon_prob: {recon_epoch}, kl: {kl_epoch}, tb: {tb_epoch}")
    
    if epoch % args.log_image_freq == 0 and args.track:
        with torch.no_grad():
            gen_samples = vae.sample(10)
            ground_truth = x[:10]
            _, _, _, recon_samples = vae(ground_truth)
        
        fig1, axs1 = plt.subplots(2, 5, figsize=(4 * 5, 8))
        for i, ax in enumerate(axs1.flatten()):
            ax.imshow(gen_samples[i].cpu().squeeze(), cmap='gray')
            ax.axis('off')
        fig2, axs2 = plt.subplots(2, 5, figsize=(4 * 5, 8))
        for i, ax in enumerate(axs2.flatten()):
            ax.imshow(ground_truth[i].cpu().squeeze(), cmap='gray')
            ax.axis('off')
        fig3, axs3 = plt.subplots(2, 5, figsize=(4 * 5, 8))
        for i, ax in enumerate(axs3.flatten()):
            ax.imshow(recon_samples[i].cpu().squeeze(), cmap='gray')
            ax.axis('off')
        wandb.log({"elbo": elbo_epoch, "kl": kl_epoch, "recon_prob": recon_epoch, "tb_loss": tb_epoch, "elbo_test": elbo_epoch_test, "kl_test": kl_epoch_test, "recon_test": recon_epoch_test, "gen_samples": wandb.Image(fig1), "ground_truth": wandb.Image(fig2), "recon_samples": wandb.Image(fig3)})
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
    elif args.track:
        wandb.log({"elbo": elbo_epoch, "kl": kl_epoch, "recon_prob": recon_epoch, "tb_loss": tb_epoch, "elbo_test": elbo_epoch_test, "kl_test": kl_epoch_test, "recon_test": recon_epoch_test})