import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, in_channels, out_channels, emb_dim=128):
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
    def __init__(self, in_channels, out_channels, emb_dim=128):
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
    def __init__(self, channels, image_size, attention_dim, text_emb_dim=128):
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
    
class UNet_Simple(nn.Module):
    def __init__(self, clip, c_in=1, c_out=1, time_dim = 128, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)
        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.text_encoder = clip

        for param in self.text_encoder.parameters():
            param.requires_grad = False

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
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        _, text_emb, _ = self.text_encoder(None, text, None)
        text_emb = text_emb.squeeze(1)
        t += text_emb

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)
        output = self.outc(x)

        return output

class UNet(nn.Module):
    def __init__(self, clip, c_in=1, c_out=1, time_dim=128, use_cross_atn = True, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.use_cross_atn = use_cross_atn

        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        if use_cross_atn: self.cross_attention1 = CrossAttention(128, 32, 128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        if use_cross_atn: self.cross_attention2 = CrossAttention(256, 16, 256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)
        if use_cross_atn: self.cross_attention3 = CrossAttention(256, 8, 256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        if use_cross_atn: self.ca4 = CrossAttention(128, 16, 128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        if use_cross_atn: self.ca5 = CrossAttention(64, 32, 64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        if use_cross_atn: self.ca6 = CrossAttention(64, 64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.text_encoder = clip

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
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        _, text_emb, _ = self.text_encoder(None, text, None)
        text_emb = text_emb.squeeze(1)
        
        seq_text_emb = None
        if self.use_cross_atn:
            _, seq_text_emb, _ = self.text_encoder(None, text, None, return_sequence=True)
            seq_text_emb = seq_text_emb.squeeze(1)

        t += text_emb

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x2 = self.cross_attention1(x2, seq_text_emb)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x3 = self.cross_attention2(x3, seq_text_emb)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        x4 = self.cross_attention3(x4, seq_text_emb)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.ca4(x, seq_text_emb)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.ca5(x, seq_text_emb)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        x = self.ca6(x, seq_text_emb)

        output = self.outc(x)

        return output