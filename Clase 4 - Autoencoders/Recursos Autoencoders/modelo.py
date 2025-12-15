from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        # usando nn.Sequential definan las capas del stacked autoencoder,
        # tanto en el encoder, como en el decoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(len(x), -1)
        # COMPLETAR AQUI
        h = self.encoder(x)
        x = self.decoder(h)
        ############################
        x = x.view(len(x), 1, 28, 28)
        return x