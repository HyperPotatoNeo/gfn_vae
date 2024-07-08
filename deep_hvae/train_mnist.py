import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb
from networks import VAE
import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100000)
parser.add_argument('--log_image_freq', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--latent_channels', type=int, default=8)
parser.add_argument('--depth', type=int, default=1)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
parser.add_argument("--wandb-project-name", type=str, default="advantage-diffusion",
    help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default='swish',
    help="the entity (team) of wandb's project")
parser.add_argument("--run-name", type=str, default="hvae-mnist")
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

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

vae = VAE(in_chan=1, out_chan=1, latent_channels=args.latent_channels, depth=args.depth, x_dist='bernoulli').to(device)
opt = optim.Adam(vae.parameters(), lr=args.lr)

for epoch in range(args.n_epochs):
    elbo_epoch = 0.0
    recon_epoch = 0.0
    kl_epoch = 0.0
    kl_list_epoch = [0.0] * args.depth
    elbo_epoch_test = 0.0
    recon_epoch_test = 0.0
    kl_epoch_test = 0.0
    kl_list_epoch_test = [0.0] * args.depth
    
    for i, (x, _) in enumerate(dataloader):
        x = x.to(device)
        x = torch.clamp(x, 0.0, 1.0)
        elbo, recon_term, kl_term, kl_list = vae.elbo(x)
        loss = -elbo
        opt.zero_grad()
        loss.backward()
        print(i, loss.item())
        opt.step()
        elbo_epoch += elbo.item()
        recon_epoch += recon_term.item()
        kl_epoch += kl_term.item()
        kl_list_epoch = [kl_list_epoch[k] + kl_list[k].item() for k in range(args.depth)]
    with torch.no_grad():
        for j, (x, _) in enumerate(testloader):
            x = x.to(device)
            x = torch.clamp(x, 0.0, 1.0)
            elbo, recon_term, kl_term, kl_list = vae.elbo(x)
            elbo_epoch_test += elbo.item()
            recon_epoch_test += recon_term.item()
            kl_epoch_test += kl_term.item()
            kl_list_epoch_test = [kl_list_epoch_test[k] + kl_list[k].item() for k in range(args.depth)]
    elbo_epoch /= i
    recon_epoch /= i
    kl_epoch /= i
    kl_list_epoch = [kl_list_epoch[k] / i for k in range(args.depth)]
    elbo_epoch_test /= j
    recon_epoch_test /= j
    kl_epoch_test /= j
    kl_list_epoch_test = [kl_list_epoch_test[k] / j for k in range(args.depth)]
    print(f"Epoch: {epoch}, ELBO: {elbo_epoch}, recon_prob: {recon_epoch}, kl: {kl_epoch}")
    
    if epoch % args.log_image_freq == 0 and args.track:
        with torch.no_grad():
            z_list, z_dist_list, log_prob_list = vae.generator(10)
            gen_samples = torch.nn.functional.sigmoid(z_dist_list[-1].logits)
            ground_truth = x[:10]
        
        fig1, axs1 = plt.subplots(2, 5, figsize=(4 * 5, 8))
        for i, ax in enumerate(axs1.flatten()):
            ax.imshow(gen_samples[i].cpu().squeeze(), cmap='gray')
            ax.axis('off')
        fig2, axs2 = plt.subplots(2, 5, figsize=(4 * 5, 8))
        for i, ax in enumerate(axs2.flatten()):
            ax.imshow(ground_truth[i].cpu().squeeze(), cmap='gray')
            ax.axis('off')
        wandb.log({"elbo": elbo_epoch, "kl": kl_epoch, "recon_prob": recon_epoch, "elbo_test": elbo_epoch_test, "kl_test": kl_epoch_test, "recon_test": recon_epoch_test, "gen_samples": wandb.Image(fig1), "ground_truth": wandb.Image(fig2)})
        plt.close(fig1)
        plt.close(fig2)
    elif args.track:
        wandb.log({"elbo": elbo_epoch, "kl": kl_epoch, "recon_prob": recon_epoch, "elbo_test": elbo_epoch_test, "kl_test": kl_epoch_test, "recon_test": recon_epoch_test})