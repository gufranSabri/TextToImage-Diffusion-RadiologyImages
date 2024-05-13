import torch
from tqdm import tqdm
import numpy as np

class Diffusion:
    def __init__(
            self, 
            noise_steps=1000, 
            beta_start=1e-4, 
            beta_end=0.02, 
            img_size=64, 
            scheduler_type='linear', 
            device="cuda"
        ):
        
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.device = device

        if scheduler_type == 'linear':
            self.beta = self.prepare_linear_noise_schedule().to(device)
        elif scheduler_type == 'cosine':
            self.beta = self.prepare_cosine_noise_schedule().to(device)
        
        self.alpha = 1. - self.beta
        self.alpha = self.alpha.to(device)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(device)

        self.img_size = img_size

    def prepare_linear_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def prepare_cosine_noise_schedule(self):
        t = torch.linspace(0, 1, self.noise_steps, device=self.device)
        cos_schedule = 0.5 * (1 + np.cos(np.pi * t))
        beta = self.beta_start + 0.5 * (self.beta_end - self.beta_start) * cos_schedule
        return beta

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]

        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample_previous_timestep(self, predicted_noise, x, t):
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]

        noise = torch.randn_like(x)
        noise = torch.where(t[:, None, None, None] < 1, torch.zeros_like(noise), noise)
    
        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        return x

    def sample(self, model, n, text):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, desc="Sampling"):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, text)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
