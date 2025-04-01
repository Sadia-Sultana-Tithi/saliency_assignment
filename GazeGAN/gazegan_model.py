import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CSCBlock(nn.Module):
    """Center-Surround Connection Block"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv_center = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        self.conv_surround = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, dilation=2, padding=2)
        
    def forward(self, x):
        center = self.conv_center(x)
        surround = self.conv_surround(x)
        return torch.cat([center, surround], dim=1)

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, use_csc=False):
        super().__init__()
        self.use_csc = use_csc
        layers = [
            nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_size),
            nn.LeakyReLU(0.2)
        ]
        if use_csc:
            layers.append(CSCBlock(out_size))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, use_csc=False):
        super().__init__()
        self.use_csc = use_csc
        layers = [
            nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True)
        ]
        if use_csc:
            layers.append(CSCBlock(out_size))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat([x, skip_input], dim=1)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, use_csc=True):
        super().__init__()
        
        self.down1 = UNetDown(in_channels, 64, use_csc)
        self.down2 = UNetDown(64, 128, use_csc)
        self.down3 = UNetDown(128, 256, use_csc)
        self.down4 = UNetDown(256, 512, use_csc)
        self.down5 = UNetDown(512, 512, use_csc)
        
        self.up1 = UNetUp(512, 512, use_csc)
        self.up2 = UNetUp(1024, 256, use_csc)
        self.up3 = UNetUp(512, 128, use_csc)
        self.up4 = UNetUp(256, 64, use_csc)
        
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, kernel_size=4, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        
        return self.final(u4)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        # Changed input channels to 4 (3 for image + 1 for saliency map)
        self.model = nn.Sequential(
            *discriminator_block(4, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )
        
    def forward(self, img_A, img_B):
        # Ensure img_B has 1 channel (saliency map)
        if img_B.shape[1] > 1:
            img_B = img_B.mean(dim=1, keepdim=True)
        img_input = torch.cat([img_A, img_B], dim=1)
        return self.model(img_input)

class HistogramLoss(nn.Module):
    def __init__(self, bins=100, min=0.0, max=1.0):
        super().__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.delta = (max - min) / bins
        
    def forward(self, pred, target):
        pred_hist = torch.zeros(self.bins, device=pred.device)
        target_hist = torch.zeros(self.bins, device=target.device)
        
        for i in range(self.bins):
            bin_val = self.min + i * self.delta
            pred_hist[i] = (pred >= bin_val).float().sum()
            target_hist[i] = (target >= bin_val).float().sum()
            
        pred_hist = pred_hist / pred_hist.sum()
        target_hist = target_hist / target_hist.sum()
        
        return F.kl_div(pred_hist.log(), target_hist, reduction='batchmean')

class GazeGAN(nn.Module):
    def __init__(self, use_csc=True, lambda_hist=0.5):
        super().__init__()
        self.generator = Generator(use_csc=use_csc)
        self.discriminator = Discriminator()
        
        # Loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_L1 = nn.L1Loss()
        self.criterion_hist = HistogramLoss()
        self.lambda_hist = lambda_hist
        
    def forward(self, x):
        return self.generator(x)
    
    def compute_generator_loss(self, real_images, real_maps, fake_maps):
        # GAN loss
        pred_fake = self.discriminator(real_images, fake_maps)
        loss_GAN = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        
        # L1 loss
        loss_L1 = self.criterion_L1(fake_maps, real_maps)
        
        # Histogram loss
        loss_hist = self.criterion_hist(fake_maps, real_maps)
        
        # Total loss
        return loss_GAN + loss_L1 + self.lambda_hist * loss_hist
    
    def compute_discriminator_loss(self, real_images, real_maps, fake_maps):
        # Real loss
        pred_real = self.discriminator(real_images, real_maps)
        loss_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))
        
        # Fake loss
        pred_fake = self.discriminator(real_images, fake_maps.detach())
        loss_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        
        return (loss_real + loss_fake) * 0.5