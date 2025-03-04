import os
import torch
import torchvision
from PIL import Image
import pandas as pd
import numpy as np

import clip

caption_mode = {
    "CAPTION": 0,
    "CAPTION_CUI": 1,
    "KEYWORDS": 2,
    "CAPTION_CUI_KEYWORDS": 3
}

class CLIP_dataset():
    def __init__(self, image_list, list_captions, preprocessor):
        self.image_list = image_list
        self.list_captions  = clip.tokenize(list_captions)

        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.list_captions)

    def __getitem__(self, idx):
        image = self.preprocessor(Image.open(self.image_list[idx]))
        list_captions = self.list_captions[idx]

        return image, list_captions

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_total_batches(data_path, phase='train', batch_size=32, top_k_cui = 20):
    df_name = f"{phase}.csv"
    if top_k_cui is not None: 
        df_name = f"{phase}_top{top_k_cui}_kcf.csv"

    df = pd.read_csv(os.path.join(data_path, "processed", df_name))

    return df.shape[0]//batch_size

def diffusion_data_generator(data_path, phase='train', image_size = 64, batch_size=32, top_k_cui = 20, shuffle=True, cm = caption_mode["CAPTION_CUI_KEYWORDS"]):
    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(image_size + 32),
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=Image.BICUBIC),
        torchvision.transforms.Lambda(lambda x: torchvision.transforms.functional.equalize(x)),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ])

    df_name = f"{phase}.csv"
    if top_k_cui is not None: 
        df_name = f"{phase}_top{top_k_cui}_kcf.csv"
    
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

            caption = df.iloc[i]['Caption']
            if cm == caption_mode["CAPTION_CUI"]: caption = df.iloc[i]['Caption'] + " " + df.iloc[i]['CUI_caption']
            if cm == caption_mode["KEYWORDS"]: caption = df.iloc[i]['keywords']
            if cm == caption_mode["CAPTION_CUI_KEYWORDS"]: caption = df.iloc[i]['keywords'] + " " +  df.iloc[i]['Caption'] + " " + df.iloc[i]['CUI_caption']
            
            captions.append(caption)

        images = torch.stack(images)
        yield images, captions

def test_diffuser(model, diffusion_model, text_tokenizer, save_path, image_size = 64, k = 20, use_clip = False, cm = caption_mode["CAPTION_CUI_KEYWORDS"], skip_amount = 0, device = "cuda"):
    model.eval()

    generator = diffusion_data_generator("./data/rocov2", phase="test", batch_size=1, image_size = image_size, top_k_cui=k, shuffle=False, cm = cm)
    count = 0
    for _, captions in generator:
        if count < skip_amount:
            count += 1
            continue

        print(count, captions)
        if not use_clip:
          for i, c in enumerate(captions):
            inputs = text_tokenizer(c, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
            captions[i] = torch.stack([inputs["input_ids"], inputs["attention_mask"]]).to(device)
        else:
          captions = clip.tokenize(captions).to(device)

        count+=1
        sampled_images = diffusion_model.sample(model, len(captions), captions)
        save_images(sampled_images, os.path.join(save_path, f"{count}.jpg"))
