import os
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from model import *
from transformers import AutoTokenizer
import argparse
import time

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(args):
    if not os.path.exists("./results"):
      os.mkdir("./results")
    
    res_path = os.path.join("./results", args.model + str(len(os.listdir("./results"))+1))
    images_path = os.path.join(res_path, "images")
    text_path = os.path.join(res_path, "text")
    model_path = os.path.join(res_path, "model")

    os.mkdir(res_path)
    os.mkdir(images_path)
    os.mkdir(text_path)
    os.mkdir(model_path)

    model = None
    if args.model == "UNet_Simple":
      model = UNet_Simple().to(args.device)
    elif args.model == "UNet":
      model = UNet().to(args.device)

    diffusion = Diffusion(img_size=args.image_size, device=args.device)
    text_tokenizer = AutoTokenizer.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    mse = nn.MSELoss()
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    total_batches = get_total_batches(args.data_path, phase='train', num_samples=int(args.num_samples), batch_size=args.batch_size)
    for epoch in range(args.epochs):
      gen = data_generator(args.data_path, phase="train", num_samples = int(args.num_samples), batch_size=args.batch_size)

      total_loss = 0
      curr_time = time.time()
      for images, captions in tqdm(gen, desc=f"Epoch {epoch+1}", total=total_batches):
        images = images.to(args.device)
        
        for i, c in enumerate(captions):
          inputs = text_tokenizer(c, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
          captions[i] = torch.stack([inputs["input_ids"], inputs["attention_mask"]]).to(args.device)
         
        t = diffusion.sample_timesteps(images.shape[0]).to(args.device)
        x_t, noise = diffusion.noise_images(images, t)
        predicted_noise = model(x_t, t, captions)

        loss = mse(noise, predicted_noise)
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        ema.step_ema(ema_model, model)

        total_loss += loss.item()

      test_gen = data_generator(args.data_path, phase="test", batch_size=4)
      captions = None
      for images, captions in test_gen:
        for i, c in enumerate(captions):
          with open(os.path.join(text_path, f"{epoch}.txt"), 'a') as f:
            f.write(c + "\n")
    
          inputs = text_tokenizer(c, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
          captions[i] = torch.stack([inputs["input_ids"], inputs["attention_mask"]]).to(args.device)
        break

      sampled_images = diffusion.sample(model, len(captions), captions)
      save_images(sampled_images, os.path.join(images_path, f"{epoch}.jpg"))

      print()
      print(f"Total Loss: {total_loss/total_batches}")
      print(f"Time taken: {time.time()-curr_time}")
      print()
    
    torch.save(ema_model.state_dict(), os.path.join(model_path, "ema_model.pth"))
      
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-data_path',dest='data_path', default='./data/rocov2')
    parser.add_argument('-model',dest='model', default="UNet_Simple")
    parser.add_argument('-num_samples',dest='num_samples', default=1000)
    parser.add_argument('-epochs',dest='epochs', default=300)
    parser.add_argument('-batch_size',dest='batch_size', default=16)
    parser.add_argument('-image_size',dest='image_size', default=64)
    parser.add_argument('-device',dest='device', default='cuda')
    args=parser.parse_args()

    train(args)