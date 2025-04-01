import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from gazegan_model import GazeGAN

class SaliencyDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'Images')
        self.td_fixation_dir = os.path.join(root_dir, 'TD_FixMaps')
        self.asd_fixation_dir = os.path.join(root_dir, 'ASD_FixMaps')
        
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.transform_gray = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        base_name = os.path.splitext(img_name)[0]
        
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        td_fix_path = os.path.join(self.td_fixation_dir, f"{base_name}_s.png")
        asd_fix_path = os.path.join(self.asd_fixation_dir, f"{base_name}_s.png")
        
        try:
            td_fix_map = Image.open(td_fix_path).convert('L')
        except:
            td_fix_map = Image.new('L', (256, 256))
            
        try:
            asd_fix_map = Image.open(asd_fix_path).convert('L')
        except:
            asd_fix_map = Image.new('L', (256, 256))
        
        image = self.transform(image)
        td_fix_map = self.transform_gray(td_fix_map)
        asd_fix_map = self.transform_gray(asd_fix_map)
        
        return {
            'image': image,
            'td_fix_map': td_fix_map,
            'asd_fix_map': asd_fix_map,
            'img_name': img_name
        }

def train():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    epochs = 10
    batch_size = 4
    lr = 0.0002
    beta1 = 0.5
    
    # Initialize dataset and dataloader
    dataset = SaliencyDataset(root_dir='C:/Users/User/Downloads/Saliency4asd/Saliency4asd')
    print(f"Found {len(dataset)} images in dataset")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = GazeGAN(use_csc=True).to(device)
    optimizer_G = optim.Adam(model.generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Training loop
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            # Move data to device
            real_images = batch['image'].to(device)
            td_fix_maps = batch['td_fix_map'].to(device)
            
            # Generate fake saliency maps
            fake_maps = model(real_images)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            loss_D = model.compute_discriminator_loss(
                real_images, 
                td_fix_maps[:, :1, :, :],
                fake_maps[:, :1, :, :]
            )
            loss_D.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            loss_G = model.compute_generator_loss(
                real_images,
                td_fix_maps[:, :1, :, :],
                fake_maps[:, :1, :, :]
            )
            loss_G.backward()
            optimizer_G.step()
            
            # Print progress
            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{i}/{len(dataloader)}], '
                      f'D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}')
    
    # Save final model
    torch.save(model.state_dict(), 'gazegan_final.pth')
    print("Training complete! Model saved as gazegan_final.pth")

if __name__ == '__main__':
    train()