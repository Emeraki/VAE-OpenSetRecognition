import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):
    def __init__(self, input_size, h_dim, z_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, input_size),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z

    def forward(self, x):
        h = self.encoder(x)

        mu, logvar = torch.chunk(h, 2, dim=-1)

        z = self.reparameterize(mu, logvar)

        return self.decoder(z), mu, logvar


def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


# if __name__ == '__main__':
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     vae = VAE(input_size=10, h_dim=6, z_dim=3)
#     vae = vae.to(device)
#
#     optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
#
#     input = torch.randn((4, 10))
#     input = input.to(device)
#     print(input)
#     print("===================================")
#
#     recon_x, mu, logvar = vae(input)
#     print(recon_x)
#     print("===================================")
#     print(mu)
#     print("===================================")
#     print(logvar)
#     print("===================================")
#
#     all_loss, BCE_loss, KLD_loss = loss_fn(recon_x, input, mu, logvar)
#     print(all_loss)
#     print(BCE_loss)
#     print(KLD_loss)
#
#     optimizer.zero_grad()
#     all_loss.backward()
#     optimizer.step()
