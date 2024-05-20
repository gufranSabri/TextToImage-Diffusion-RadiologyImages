from tqdm import tqdm
import argparse
import pandas as pd
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip

from utils.utils import CLIP_dataset

def main(args):
    device = args.device
    
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model = model.to(device)

    df = pd.read_csv(f"{args.data_path}/processed/{args.phase}_top_20_key_cf.csv")

    images = df["ID"].tolist()
    captions = df["keywords"].tolist()
    images = [os.path.join(args.data_path, "train_images", f"{img}.jpg") for img in images]

    dataset = CLIP_dataset(images, captions, preprocess)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9,0.98), eps=1e-6, weight_decay=0.2)
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    patience = 7
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, total=len(dataloader), desc = f"Epoch {epoch+1}")

        count, total = 0, 0
        for batch in pbar:
            optimizer.zero_grad()

            images,texts = batch 
            images= images.to(device)
            texts = texts.to(device)

            logits_per_image, logits_per_text = model(images, texts)

            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            total += total_loss.item()
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            count+=1
        
        print(f"Loss: {(total/count):.4f} | Patience: {patience}\n")
        if total/count < best_val_loss:
            best_val_loss = total/count
            patience = 7
        else:
            patience -= 1

        if patience == 0:
            print("=========================================\nEarly stopping\n=========================================\n")
            break

    torch.save({
        'model_state_dict': model.state_dict(),
    }, './models/clip.pth')

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-data_path',dest='data_path', default='./data/rocov2')
    parser.add_argument('-epochs',dest='epochs', default=500)
    parser.add_argument('-phase',dest='phase', default='train')
    parser.add_argument('-device',dest='device', default='cuda')
    args=parser.parse_args()

    main(args)