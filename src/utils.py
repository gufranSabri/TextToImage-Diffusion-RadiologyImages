import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_total_batches(data_path, phase='train', num_samples=None, batch_size=32):
    df = pd.read_csv(os.path.join(data_path, f"{phase}.csv"))

    if num_samples is not None: return num_samples//batch_size
    return df.shape[0]//batch_size

def data_generator(data_path, phase='train', image_size = 64, num_samples=None, batch_size=32):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=Image.BICUBIC),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    df = pd.read_csv(os.path.join(data_path, f"{phase}.csv"))
    df = df.sample(frac=1).reset_index(drop=True)
    if num_samples is not None: df = df.sample(num_samples)
    else: num_samples = df.shape[0]
    
    for b in range(num_samples//batch_size):
        images, captions = [], []
        for i in range(b*batch_size, (b+1)*batch_size):
            img_path = os.path.join(data_path, f"{phase}_images",f"{df.iloc[i]['ID']}.jpg")
            img = Image.open(img_path)
            img = transforms(img)
            images.append(img)

            caption = df.iloc[i]['Caption'] + " " + df.iloc[i]['CUI_caption']
            captions.append(caption)
        
        images = torch.stack(images)
        yield images, captions

if __name__ == "__main__":
    gen = data_generator("./data/rocov2")
    phase = 'train'
    batch_size = 32
    data_path = "./data/rocov2"

    total = pd.read_csv(os.path.join(data_path, f"{phase}.csv")).shape[0]/batch_size
    for images, captions in tqdm(gen, total=total):
        print(images.shape, len(captions))
        break