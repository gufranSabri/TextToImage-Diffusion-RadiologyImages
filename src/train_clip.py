import os
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils.utils import *
from transformers import AutoTokenizer
import argparse
import time

from clip.mtl_clip import *

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(args):
    if not os.path.exists("./models"):
      os.mkdir("./models")

    model = MTL_CLIP().to(args.device)
    text_tokenizer = AutoTokenizer.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    patience = 7
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        total_batches = get_total_batches(args.data_path, phase='train', batch_size=args.batch_size)
        gen = diffusion_data_generator(args.data_path, phase="train", img_channels=3, return_cui = True, batch_size=args.batch_size)

        model.train()
        total_loss = 0
        for images, captions, cuis, cui_captions in tqdm(gen, desc=f"Epoch {epoch+1}", total=total_batches):
            images = images.to(args.device)

            cuis_set = []
            cui_captions_set = []
            img_cui_map = []
            for i in range(len(cuis)):
                temp_cuis = cuis[i].split(";")
                img_cui_map.append([])
                for j, tc in enumerate(temp_cuis):
                    if tc not in cuis_set:
                        cuis_set.append(tc)
                        cui_captions_set.append(cui_captions[i].split(";")[j])
                        img_cui_map[-1].append(len(cuis_set)-1)
                    else:
                        img_cui_map[-1].append(cuis_set.index(tc))

            caption_labels = torch.zeros((args.batch_size, args.batch_size))
            for i in range(args.batch_size):
                for j in range(args.batch_size):
                    if i == j:
                        caption_labels[i][j] = 1
            
            cui_labels = torch.zeros((args.batch_size, len(cuis_set)))
            for i in range(args.batch_size):
                for j in img_cui_map[i]:
                    cui_labels[i][j] = 1            

            for i in range(len(captions)):
                inputs_captions = text_tokenizer(captions[i], return_tensors="pt", max_length=64, truncation=True, padding="max_length")
                captions[i] = torch.stack([inputs_captions["input_ids"], inputs_captions["attention_mask"]]).to(args.device)

            for i in range(len(cui_captions_set)):
                inputs_cuis = text_tokenizer(cui_captions_set[i], return_tensors="pt", max_length=64, truncation=True, padding="max_length")
                cui_captions_set[i] = torch.stack([inputs_cuis["input_ids"], inputs_cuis["attention_mask"]]).to(args.device)

            image_emb, caption_emb, cui_emb = model(images, captions, cui_captions_set)
            caption_loss, cui_loss, _, _ = model.compute_loss(image_emb, caption_emb, cui_emb, caption_labels, cui_labels)

            loss = caption_loss + cui_loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        valid_gen = diffusion_data_generator(args.data_path, phase="valid", img_channels=3, return_cui = True, batch_size=args.batch_size)
        val_total_batches = get_total_batches(args.data_path, phase='valid', batch_size=args.batch_size)
        val_loss = 0

        model.eval()
        for images, captions, cuis, cui_captions in tqdm(valid_gen, desc=f"Validation {epoch+1}", total=val_total_batches):
            images = images.to(args.device)

            cuis_set = []
            cui_captions_set = []
            img_cui_map = []
            for i in range(len(cuis)):
                temp_cuis = cuis[i].split(";")
                img_cui_map.append([])
                for j, tc in enumerate(temp_cuis):
                    if tc not in cuis_set:
                        cuis_set.append(tc)
                        cui_captions_set.append(cui_captions[i].split(";")[j])
                        img_cui_map[-1].append(len(cuis_set)-1)
                    else:
                        img_cui_map[-1].append(cuis_set.index(tc))

            caption_labels = torch.zeros((args.batch_size, args.batch_size))
            for i in range(args.batch_size):
                for j in range(args.batch_size):
                    if i == j:
                        caption_labels[i][j] = 1
            
            cui_labels = torch.zeros((args.batch_size, len(cuis_set)))
            for i in range(args.batch_size):
                for j in img_cui_map[i]:
                    cui_labels[i][j] = 1            

            for i in range(len(captions)):
                inputs_captions = text_tokenizer(captions[i], return_tensors="pt", max_length=64, truncation=True, padding="max_length")
                captions[i] = torch.stack([inputs_captions["input_ids"], inputs_captions["attention_mask"]]).to(args.device)

            for i in range(len(cui_captions_set)):
                inputs_cuis = text_tokenizer(cui_captions_set[i], return_tensors="pt", max_length=64, truncation=True, padding="max_length")
                cui_captions_set[i] = torch.stack([inputs_cuis["input_ids"], inputs_cuis["attention_mask"]]).to(args.device)

            image_emb, caption_emb, cui_emb = model(images, captions, cui_captions_set)
            caption_loss, cui_loss, _, _ = model.compute_loss(image_emb, caption_emb, cui_emb, caption_labels, cui_labels)

            loss = caption_loss + cui_loss
            val_loss += loss.item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 7
        else:
            patience -= 1

        print(f"Training Loss: {total_loss/total_batches}")
        print(f"Validation Loss: {val_loss/val_total_batches}")
        print(f"Patience: {patience}")
        print()
        
        if patience == 0:
            print("Early Stopping")
            break
        
    torch.save(model.state_dict(), f"./models/mtl_clip.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default="./data/rocov2")
    parser.add_argument("-device", type=str, default="cuda")
    parser.add_argument("-epochs", type=int, default=50)
    parser.add_argument("-batch_size", type=int, default=16)
    args = parser.parse_args()

    train(args)