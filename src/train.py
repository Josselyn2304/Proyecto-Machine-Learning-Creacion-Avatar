import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import glob
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.ugatit import UGATIT

class CelebADataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        self.transform = transform
        print(f'Dataset: {len(self.image_paths)} imagenes')
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, image
        except:
            return torch.zeros(3, 256, 256), torch.zeros(3, 256, 256)

class Trainer:
    def __init__(self, config_path='config.yaml'):
        print('\n' + '='*70)
        print('ENTRENAMIENTO U-GAT-IT')
        print('='*70 + '\n')
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Dispositivo: {self.device}')
        
        if torch.cuda.is_available():
            print(f'GPU: {torch.cuda.get_device_name(0)}')
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f'VRAM: {vram:.2f} GB\n')
        
        self.model = UGATIT(style_dim=256).to(self.device)
        print('Modelo creado\n')
        
        lr = self.config['training']['lr']
        self.G_optim = torch.optim.Adam(
            list(self.model.genA2B.parameters()) + list(self.model.genB2A.parameters()),
            lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.D_optim = torch.optim.Adam(
            list(self.model.disGA.parameters()) + list(self.model.disGB.parameters()) +
            list(self.model.disLA.parameters()) + list(self.model.disLB.parameters()),
            lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.epoch = 0
        self.best_loss = float('inf')
    
    def create_dataloader(self):
        img_size = self.config['model']['img_size']
        batch_size = self.config['training']['batch_size']
        
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
        dataset = CelebADataset('data/celeba/processed/train', transform)
        return DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    
    def train_epoch(self, dataloader):
        self.model.genA2B.train()
        self.model.genB2A.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0
        
        for real_A, real_B in tqdm(dataloader, desc=f'Epoca {self.epoch+1}'): 
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)
            
            self.G_optim.zero_grad()
            fake_B = self.model.genA2B(real_A)
            fake_A = self.model.genB2A(real_B)
            recon_A = self.model.genB2A(fake_B)
            recon_B = self.model.genA2B(fake_A)
            
            cycle_loss = self.L1_loss(recon_A, real_A) + self.L1_loss(recon_B, real_B)
            identity_A = self.model.genB2A(real_A)
            identity_B = self.model.genA2B(real_B)
            identity_loss = self.L1_loss(identity_A, real_A) + self.L1_loss(identity_B, real_B)
            
            G_loss = 10.0 * cycle_loss + 5.0 * identity_loss
            G_loss.backward()
            self.G_optim.step()
            
            self.D_optim.zero_grad()
            real_logit = self.model.disGB(real_B)
            fake_logit = self.model.disGB(fake_B.detach())
            D_real = self.MSE_loss(real_logit, torch.ones_like(real_logit))
            D_fake = self.MSE_loss(fake_logit, torch.zeros_like(fake_logit))
            D_loss = (D_real + D_fake) * 0.5
            D_loss.backward()
            self.D_optim.step()
            
            epoch_g_loss += G_loss.item()
            epoch_d_loss += D_loss.item()
            num_batches += 1
            
            if num_batches % 50 == 0:
                torch.cuda.empty_cache()
        
        avg_g = epoch_g_loss / num_batches
        avg_d = epoch_d_loss / num_batches
        return avg_g, avg_d
    
    def save_checkpoint(self, filename=None):
        os.makedirs('models/checkpoints', exist_ok=True)
        if filename is None:
            filename = f'epoch_{self.epoch+1}.pth'
        filepath = os.path.join('models/checkpoints', filename)
        torch.save({
            'epoch': self.epoch+1,
            'genA2B': self.model.genA2B.state_dict(),
            'genB2A': self.model.genB2A.state_dict()
        }, filepath)
        print(f'Checkpoint: {filepath}')
    
    def train(self):
        num_epochs = self.config['training']['num_epochs']
        batch_size = self.config['training']['batch_size']
        print(f'Epocas: {num_epochs}, Batch: {batch_size}\n')
        
        dataloader = self.create_dataloader()
        start = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            g_loss, d_loss = self.train_epoch(dataloader)
            
            elapsed = time.time() - start
            mins = elapsed / 60
            print(f'\nEpoca {epoch+1}: G={g_loss:.4f}, D={d_loss:.4f}, Tiempo={mins:.1f}min')
            
            if (epoch+1) % 5 == 0:
                self.save_checkpoint()
            if g_loss < self.best_loss:
                self.best_loss = g_loss
                self.save_checkpoint('best.pth')
        
        total_hours = (time.time() - start) / 3600
        print(f'\nCompletado: {total_hours:.2f} horas\n')

if __name__ == '__main__':
    Trainer().train()