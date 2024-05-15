import os
import torch
import torchvision
from PIL import Image
import pandas as pd

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_total_batches(data_path, phase='train', batch_size=32, top_k_cui = 20, use_transformed_caption = True):
    df_name = f"{phase}.csv"
    if top_k_cui is not None: 
        if use_transformed_caption: df_name = f"{phase}_top_{top_k_cui}_key_cf_knee.csv"
        else: df_name = f"{phase}_top_{top_k_cui}_cui.csv"

    df = pd.read_csv(os.path.join(data_path, "processed", df_name))
    # df = df.sample(n=992).reset_index(drop=True)
    
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
    
    # df = df.sample(n=992).reset_index(drop=True)
    
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

def diffusion_data_generator(data_path, phase='train', return_cui = False, img_channels = 1, image_size = 64, batch_size=32, top_k_cui = 20, use_transformed_caption = True):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size + 32),
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=Image.BICUBIC),
        torchvision.transforms.Grayscale(num_output_channels=img_channels),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    df_name = f"{phase}.csv"
    if top_k_cui is not None: 
        if use_transformed_caption: df_name = f"{phase}_top_{top_k_cui}_key_cf_knee.csv"
        else: df_name = f"{phase}_top_{top_k_cui}_cui.csv"
    
    df = pd.read_csv(os.path.join(data_path, "processed", df_name))
    df = df.sample(frac=1).reset_index(drop=True)
    
    remainder = df.shape[0] % batch_size
    if remainder > 0:
        df = df.iloc[:-remainder]
    
    for b in range(df.shape[0]//batch_size):
        images, captions, cuis, cui_captions = [], [], [], []
        for i in range(b*batch_size, (b+1)*batch_size):
            image_extension = ".jpg" if "ROCO" in df.iloc[i]['ID'] == 3 else ".png"
            img_path = os.path.join(data_path.replace("/processed", ""), f"{phase}_images",f"{df.iloc[i]['ID']}{image_extension}")
            img = Image.open(img_path)
            img = transforms(img)
            images.append(img)

            caption = df.iloc[i]['Caption'] + " " + df.iloc[i]['CUI_caption']
            if use_transformed_caption:
                caption = df.iloc[i]['transformed_caption']
            
            captions.append(caption)

            if return_cui: 
                cuis.append(df.iloc[i]['CUIs'])
                cui_captions.append(df.iloc[i]['CUI_caption'])

        images = torch.stack(images)
        if not return_cui:
            yield images, captions
        else:
            yield images, captions, cuis, cui_captions