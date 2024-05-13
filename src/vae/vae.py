import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import warnings
from utils.lpips import LPIPS
import os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class VAELoss(nn.Module):
    def __init__(self, device="cuda"):
        super(VAELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.lpips_model = LPIPS(device=device)

    def forward(self, recon_x, x, mu, log_var):
        reconstruction_loss = self.mse_loss(recon_x, x)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        lpips_loss = self.lpips_model(recon_x, x)

        loss = reconstruction_loss + kl_divergence + lpips_loss.mean()

        return loss

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(8*8*128, 64)
        self.fc_var = nn.Linear(8*8*128, 64)

        self.pre_decoder = nn.Linear(64, 8*8*128)        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # 64x64 -> 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),  # 64x64 -> 128x128
            nn.Tanh(),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        z = self.pre_decoder(z)
        z = z.view(-1, 128, 8, 8)
        return self.decoder(z)
    
    def perceptual_output(self, x):
        h = x
        h = self.slice1(h)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

if __name__ == "__main__":
    batch_size = 1
    dummy_input = torch.randn(batch_size, 1, 64, 64)

    vae = VAE()

    output, mu, log_var = vae(dummy_input)
    print(output.shape)