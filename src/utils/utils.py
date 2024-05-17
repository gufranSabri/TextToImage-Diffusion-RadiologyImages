import os
import torch
import torchvision
from PIL import Image
import pandas as pd
import numpy as np

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_total_batches(data_path, phase='train', batch_size=32, top_k_cui = 20, use_transformed_caption = True):
    df_name = f"{phase}.csv"
    if top_k_cui is not None: 
        df_name = f"{phase}_top_{top_k_cui}_key_cf.csv"

    df = pd.read_csv(os.path.join(data_path, "processed", df_name))

    return df.shape[0]//batch_size

def vae_data_generator(data_path, phase='train', base_image_size = 64, y_image_size = 256, batch_size=32):
    base_image_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),
        torchvision.transforms.RandomResizedCrop(base_image_size, scale=(0.8, 1.0), interpolation=Image.BICUBIC),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    y_image_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(y_image_size + 32),
        torchvision.transforms.RandomResizedCrop(y_image_size, scale=(0.8, 1.0), interpolation=Image.BICUBIC),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    df_name = f"{phase}.csv"
    df = pd.read_csv(os.path.join(data_path, "processed", df_name))
    df = df.sample(frac=1).reset_index(drop=True)

    remainder = df.shape[0] % batch_size
    if remainder > 0:
        df = df.iloc[:-remainder]
    
    for b in range(df.shape[0]//batch_size):
        base_images, y_images = [], []
        for i in range(b*batch_size, (b+1)*batch_size):
            base_img_path = os.path.join(data_path.replace("/processed", ""), f"{phase}_images",f"{df.iloc[i]['ID']}.jpg")
            base_img = Image.open(base_img_path)
            base_img = base_image_transform(base_img)
            base_images.append(base_img)

            y_img_path = os.path.join(data_path.replace("/processed", ""), f"{phase}_images",f"{df.iloc[i]['ID']}.jpg")
            y_img = Image.open(y_img_path)
            y_img = y_image_transform(y_img)
            y_images.append(y_img)

        base_images = torch.stack(base_images)
        y_images = torch.stack(y_images)

        yield base_images, y_images

def diffusion_data_generator(data_path, phase='train', image_size = 64, batch_size=32, top_k_cui = 20, use_keywords = True, shuffle=True):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size + 32),
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=Image.BICUBIC),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    df_name = f"{phase}.csv"
    if top_k_cui is not None: 
        df_name = f"{phase}_top_{top_k_cui}_key_cf.csv"
    
    df = pd.read_csv(os.path.join(data_path, "processed", df_name))
    if shuffle: df = df.sample(frac=1).reset_index(drop=True)
    
    remainder = df.shape[0] % batch_size
    if remainder > 0:
        df = df.iloc[:-remainder]
    
    for b in range(df.shape[0]//batch_size):
        images, captions= [], []
        for i in range(b*batch_size, (b+1)*batch_size):
            image_extension = ".jpg" if "ROCO" in df.iloc[i]['ID'] else ".png"
            img_path = os.path.join(data_path.replace("/processed", ""), f"{phase}_images",f"{df.iloc[i]['ID']}{image_extension}")
            img = Image.open(img_path)
            img = transforms(img)
            images.append(img)

            caption = df.iloc[i]['Caption'] + " " + df.iloc[i]['CUI_caption']
            if use_keywords: caption = df.iloc[i]['keywords']
            
            captions.append(caption)

        images = torch.stack(images)
        yield images, captions

def test_diffuser(model, diffusion_model, text_tokenizer, save_path, image_size = 64, k = 20, device = "cuda"):
    model.eval()

    generator = diffusion_data_generator("./data/rocov2", phase="test", batch_size=1, image_size = image_size, top_k_cui=k, shuffle=False)
    count = 0
    for _, captions in generator:
        for i, c in enumerate(captions):
            inputs = text_tokenizer(c, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
            captions[i] = torch.stack([inputs["input_ids"], inputs["attention_mask"]]).to(device)

        count+=1
        sampled_images = diffusion_model.sample(model, len(captions), captions)
        save_images(sampled_images, os.path.join(save_path, f"{count}.jpg"))
