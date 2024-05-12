import torch
from torchvision import models
from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F
import os

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MTL_CLIP(nn.Module):
    def __init__(self):
        super(MTL_CLIP, self).__init__()

        self.image_encoder = models.mobilenet_v2(pretrained=True)
        self.image_encoder.classifier = nn.Identity()
        self.image_projection = nn.Linear(1280, 128)

        self.text_encoder = AutoModel.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, 128)

    def forward(self, images, captions, cuis, return_sequence = False):
        caption_emb = []
        if captions is not None:
            for i in range(len(captions)):
                if return_sequence:
                    caption_emb.append(self.text_encoder(input_ids = captions[i][0], attention_mask = captions[i][1]).last_hidden_state)
                else:
                    caption_emb.append(self.text_encoder(input_ids = captions[i][0], attention_mask = captions[i][1]).last_hidden_state[:, 0, :])

            caption_emb = torch.stack(caption_emb)
            caption_emb = self.text_projection(caption_emb)

        cui_emb = []
        if cuis is not None:
            for i in range(len(cuis)):
                if return_sequence:
                    cui_emb.append(self.text_encoder(input_ids = cuis[i][0], attention_mask = cuis[i][1]).last_hidden_state)
                else:
                    cui_emb.append(self.text_encoder(input_ids = cuis[i][0], attention_mask = cuis[i][1]).last_hidden_state[:, 0, :])

            cui_emb = torch.stack(cui_emb)
            cui_emb = self.text_projection(cui_emb)

        image_emb = []
        if images is not None:
            image_features = self.image_encoder(images)
            image_emb = self.image_projection(image_features)
        
        return image_emb, caption_emb, cui_emb
    
    def compute_loss(self, image_embeddings, caption_embeddings, cui_embeddings, caption_labels, cui_labels):
        caption_embeddings = caption_embeddings.squeeze(1)
        cui_embeddings = cui_embeddings.squeeze(1)

        image_embeddings = F.normalize(image_embeddings, dim=1)
        caption_embeddings = F.normalize(caption_embeddings, dim=1)
        cui_embeddings = F.normalize(cui_embeddings, dim=1)

        batch_size = image_embeddings.size(0)
        caption_similarities = torch.zeros((batch_size, batch_size))
        cui_similarities = torch.zeros((batch_size, len(cui_labels[0])))

        for i in range(batch_size):
            for j in range(batch_size):
                image_embed = image_embeddings[i]
                caption_embed = caption_embeddings[j]
                
                caption_similarity = F.cosine_similarity(image_embed.unsqueeze(0), caption_embed.unsqueeze(0))
                caption_similarity = torch.clamp(caption_similarity, min=0, max=1)

                caption_similarities[i][j] = caption_similarity
        
        for i in range(batch_size):
            for j in range(len(cui_embeddings)):
                image_embed = image_embeddings[i]
                cui_embed = cui_embeddings[j]

                cui_similarity = F.cosine_similarity(image_embed.unsqueeze(0), cui_embed.unsqueeze(0))
                cui_similarity = torch.clamp(cui_similarity, min=0, max=1)

                cui_similarities[i][j] = cui_similarity

        caption_loss = F.binary_cross_entropy(caption_similarities, caption_labels, reduction='none')
        cui_loss = F.binary_cross_entropy(cui_similarities, cui_labels, reduction='none')

        caption_loss = caption_loss.mean()
        cui_loss = cui_loss.mean()
        
        return caption_loss, cui_loss, caption_similarities, cui_similarities
    
    def predict_cui(self, cui_similarities):
        return torch.argmax(cui_similarities, dim=1)