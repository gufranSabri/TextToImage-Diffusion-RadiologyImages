import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import clip
import os
import torchvision

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
class CrossAttention(nn.Module):
    def __init__(self, channels, image_size, attention_dim, text_emb_dim=256):
        super(CrossAttention, self).__init__()
        self.channels = channels
        self.attention_dim = attention_dim
        self.image_size = image_size

        self.text_projector = nn.Linear(text_emb_dim, attention_dim)
        self.mha = nn.MultiheadAttention(attention_dim, 4, batch_first=True)
        self.ln = nn.LayerNorm([attention_dim])
        self.ff = nn.Sequential(
            nn.LayerNorm([attention_dim]),
            nn.Linear(attention_dim, attention_dim),
            nn.GELU(),
            nn.Linear(attention_dim, attention_dim),
        )

    def forward(self, x, seq_text_emb):
        x = x.view(-1, self.channels, self.image_size * self.image_size).swapaxes(1, 2)
        x_ln = self.ln(x)
        
        seq_text_emb = self.text_projector(seq_text_emb)    
        seq_text_emb_ln = self.ln(seq_text_emb)    

        attention_value, _ = self.mha(x_ln, seq_text_emb_ln, seq_text_emb_ln)
        attention_output = attention_value + x
        attention_output = self.ff(attention_output) + attention_output
        
        return attention_output.swapaxes(2, 1).view(-1, self.channels, self.image_size, self.image_size)

class SelfAttention(nn.Module):
    def __init__(self, channels, image_size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.image_size = image_size

        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.image_size * self.image_size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.image_size, self.image_size)
    
class TextEncoder(nn.Module):
    def __init__(self, proj_dim=256):
        super(TextEncoder, self).__init__()
        self.text_dim = proj_dim
        
        self.text_encoder = AutoModel.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')
        self.projector = nn.Linear(768, proj_dim)
    
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def forward(self, captions, return_sequence=False):
        caption_emb = []
        for i in range(len(captions)):
            if return_sequence:
                caption_emb.append(self.text_encoder(input_ids = captions[i][0], attention_mask = captions[i][1]).last_hidden_state)
            else:
                caption_emb.append(self.text_encoder(input_ids = captions[i][0], attention_mask = captions[i][1]).last_hidden_state[:, 0, :])

        caption_emb = torch.stack(caption_emb)
        caption_emb = self.projector(caption_emb)

        return caption_emb

class ClipTextEncoder(nn.Module):
    def __init__(self, proj_dim=256, device = "cuda"):
        super(ClipTextEncoder, self).__init__()
        self.text_dim = proj_dim

        model, _ = clip.load("ViT-B/32", jit=False)
        if os.path.exists('./models/clip.pth'):
            checkpoint = torch.load('./models/clip.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            
        self.text_encoder = model
        self.projector = nn.Linear(512, proj_dim)

        self.device = device
    
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def forward(self, captions, return_sequence=False):
        caption_emb = self.text_encoder.encode_text(captions).to(torch.float32)
        caption_emb = self.projector(caption_emb)

        if not return_sequence:
            return caption_emb
        else:
            caption_emb = caption_emb.unsqueeze(1).repeat(1, 64, 1)
            return caption_emb.repeat(1, 64, 1)
        
        #     for j in range(len(captions[0])):
        #         token = torch.tensor([[captions[i][j]]]).to(self.device)
        #         token_emb = self.text_encoder.encode_text(token)
        #         sent_emb.append(token_emb)

        #     sent_emb = torch.stack(sent_emb)
        #     sent_emb = sent_emb.squeeze(1)

        #     caption_emb.append(sent_emb)

        # caption_emb = torch.stack(caption_emb).to(torch.float32)
        # caption_emb = self.projector(caption_emb)

        # return caption_emb

class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, image_size=64, use_clip = False, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, image_size//2)
        self.cross_attention1 = CrossAttention(128, image_size//2, 128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, image_size//4)
        self.cross_attention2 = CrossAttention(256, image_size//4, 256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, image_size//8)
        self.cross_attention3 = CrossAttention(256, image_size//8, 256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)
        
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, image_size//4)
        self.ca4 = CrossAttention(128, image_size//4, 128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, image_size//2)
        self.ca5 = CrossAttention(64, image_size//2, 64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, image_size)
        self.ca6 = CrossAttention(64, image_size, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if use_clip:
            print("Initializing CLIP Text Encoder")
            self.text_encoder = ClipTextEncoder(device = device)
        else:
            print("Initializing RadBERT Text Encoder")
            self.text_encoder = TextEncoder()
        
        print("Initialized VGG_UNet")

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, text):
        text_emb, seq_text_emb = None, None

        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if text is not None:
            text_emb= self.text_encoder(text)
            seq_text_emb = self.text_encoder(text, return_sequence=True)
            text_emb = text_emb.squeeze(1)
            seq_text_emb = seq_text_emb.squeeze(1)

            t += text_emb

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        if text is not None: x2 = self.cross_attention1(x2, seq_text_emb)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        if text is not None: x3 = self.cross_attention2(x3, seq_text_emb)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        if text is not None: x4 = self.cross_attention3(x4, seq_text_emb)
        
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        if text is not None: x = self.ca4(x, seq_text_emb)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        if text is not None: x = self.ca5(x, seq_text_emb)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        if text is not None: x = self.ca6(x, seq_text_emb)

        output = self.outc(x)

        return output
    

class VGG16_Unet(torch.nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, image_size=64, use_clip = False, device="cuda"):
        super(VGG16_Unet, self).__init__()

        self.device = device
        self.time_dim = time_dim

        vgg_pretrained_features = torchvision.models.vgg16(weights = "VGG16_Weights.DEFAULT").features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        
        self.inc = DoubleConv(c_in, 3)
        self.sa1 = SelfAttention(128, image_size//2)
        self.cross_attention1 = CrossAttention(128, image_size//2, 128)
        self.sa2 = SelfAttention(256, image_size//4)
        self.cross_attention2 = CrossAttention(256, image_size//4, 256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, image_size//8)
        self.cross_attention3 = CrossAttention(256, image_size//8, 256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)
        
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, image_size//4)
        self.ca4 = CrossAttention(128, image_size//4, 128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, image_size//2)
        self.ca5 = CrossAttention(64, image_size//2, 64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, image_size)
        self.ca6 = CrossAttention(64, image_size, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if use_clip:
            print("Initializing CLIP Text Encoder")
            self.text_encoder = ClipTextEncoder(device = device)
        else:
            print("Initializing RadBERT Text Encoder")
            self.text_encoder = TextEncoder()
        
        self.freeze_slices()

        print("Initialized VGG_UNet")

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def freeze_slices(self):
        print("\nFreezing Slices\n")
        for param in self.slice1.parameters():
            param.requires_grad = False
        for param in self.slice2.parameters():
            param.requires_grad = False
        for param in self.slice3.parameters():
            param.requires_grad = False
    
    def unfreeze_slices(self):
        print("\nUnfreezing Slices\n")
        for param in self.slice1.parameters():
            param.requires_grad = True
        for param in self.slice2.parameters():
            param.requires_grad = True
        for param in self.slice3.parameters():
            param.requires_grad = True

    def forward(self, x, t, text):
        text_emb, seq_text_emb = None, None

        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if text is not None:
            text_emb= self.text_encoder(text)
            seq_text_emb = self.text_encoder(text, return_sequence=True)
            text_emb = text_emb.squeeze(1)
            seq_text_emb = seq_text_emb.squeeze(1)

            t += text_emb

        x1 = self.inc(x)
        x1 = self.slice1(x1)
        x2 = self.slice2(x1)
        x2 = self.sa1(x2)
        if text is not None: x2 = self.cross_attention1(x2, seq_text_emb)
        x3 = self.slice3(x2)
        x3 = self.sa2(x3)
        if text is not None: x3 = self.cross_attention2(x3, seq_text_emb)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        if text is not None: x4 = self.cross_attention3(x4, seq_text_emb)
        
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        if text is not None: x = self.ca4(x, seq_text_emb)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        if text is not None: x = self.ca5(x, seq_text_emb)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        if text is not None: x = self.ca6(x, seq_text_emb)

        output = self.outc(x)

        return output

if __name__ == "__main__":
    from diffusion import Diffusion

    diffusion = Diffusion(device = "cpu")
    t = diffusion.sample_timesteps(1)

    model1 = VGG16_Unet(device="cpu")
    model2 = UNet(device = "cpu")

    x = torch.randn(1, 3, 64, 64)
    output = model1(x, t, None)
    print(output.shape)

    print()

    x = torch.randn(1, 1, 64, 64)
    output = model2(x, t, None)