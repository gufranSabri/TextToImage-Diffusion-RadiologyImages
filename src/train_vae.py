import torch
import torch.optim as optim
from vae.vae import VAE, VAELoss
from utils import vae_data_generator, save_images, get_total_batches
from tqdm import tqdm
import os
import datetime

def train(args):
    if not os.path.exists("./models"):os.mkdir("./models")
    
    res_path = os.path.join("./results", f"vae_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}")
    os.mkdir(res_path)

    vae = VAE().to(args.device)
    vae_loss = VAELoss(args.device).to(args.device)
    optimizer = optim.AdamW(vae.parameters(), lr=0.01)

    for epoch in range(args.num_epochs):
        data_gen = vae_data_generator(args.data_path, batch_size=args.batch_size)
        total_batches = get_total_batches(args.data_path, phase='train', batch_size=args.batch_size, top_k_cui=None)

        total_loss = 0
        for base_images, y in tqdm(data_gen, desc=f"Epoch {epoch+1}", total=total_batches):
            base_images = base_images.to(args.device)
            y = y.to(args.device)

            optimizer.zero_grad()
            outputs, mu, log_var = vae(base_images)
            loss = vae_loss(outputs, y, mu, log_var)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        val_gen = vae_data_generator(args.data_path, phase="valid", batch_size=1)
        val_total_batches = get_total_batches(args.data_path, phase='valid', batch_size=1, top_k_cui=None)
        total_val_loss = 0
        count = 0
        for base_images, y in tqdm(val_gen, total=val_total_batches, desc="Validation"):
            base_images = base_images.to(args.device)
            y = y.to(args.device)

            outputs, mu, log_var = vae(base_images)
            loss = vae_loss(outputs, y, mu, log_var)
            total_val_loss += loss.item()

            count += 1
            if count < 3:
                outputs = (outputs.clamp(-1, 1) + 1) / 2
                outputs = (outputs * 255).type(torch.uint8)
                save_images(outputs, os.path.join(res_path, f"output_{epoch}_{count}.png"))

        print(f"Training Loss: {total_loss/total_batches} Val Loss: {total_val_loss/val_total_batches}")
        print()

    torch.save(vae.state_dict(), "./models/sr_vae.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default="./data/rocov2")
    parser.add_argument("-device", type=str, default="cuda")
    parser.add_argument("-batch_size", type=int, default=16)
    parser.add_argument("-num_epochs", type=int, default=10)
    args = parser.parse_args()

    train(args)