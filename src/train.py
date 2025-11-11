import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import glob
import time

# Dataset class for CelebA
class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.file_paths = glob.glob(os.path.join(root, '*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Trainer class
class Trainer:
    def __init__(self, generator, discriminator, dataloader, criterion, optimizer_g, optimizer_d, device):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.device = device

    def train(self, epochs):
        for epoch in range(epochs):
            for i, imgs in enumerate(tqdm(self.dataloader)):
                # ------- Training the Discriminator -------
                self.optimizer_d.zero_grad()
                real_imgs = imgs.to(self.device)
                z = torch.randn(real_imgs.size(0), 100, 1, 1).to(self.device)  # Example noise input
                fake_imgs = self.generator(z)
                loss_real = self.criterion(self.discriminator(real_imgs), torch.ones(real_imgs.size(0), 1).to(self.device))
                loss_fake = self.criterion(self.discriminator(fake_imgs.detach()), torch.zeros(real_imgs.size(0), 1).to(self.device))
                d_loss = loss_real + loss_fake
                d_loss.backward()
                self.optimizer_d.step()

                # ------- Training the Generator -------
                self.optimizer_g.zero_grad()
                g_loss = self.criterion(self.discriminator(fake_imgs), torch.ones(real_imgs.size(0), 1).to(self.device))
                g_loss.backward()
                self.optimizer_g.step()

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
                print(f'Epoch [{epoch + 1}/{epochs}] - D Loss: {d_loss.item()} - G Loss: {g_loss.item()}')

    def save_checkpoint(self, epoch):
        checkpoint_path = f'checkpoint_epoch_{epoch + 1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer_g.state_dict(),
        }, checkpoint_path)
        print(f'Checkpoint saved at {checkpoint_path}')

# Main execution
if __name__ == '__main__':
    # Load configuration
    config = yaml.safe_load(open('config.yaml'))
    dataset = CelebADataset(root=config['dataset_path'], 
                            transform=transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                            ]))
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = UGATIT()  # Ensure UGATIT is properly imported
    discriminator = UGATIT()  # Similarly for the discriminator
    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config['learning_rate'])
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config['learning_rate'])

    trainer = Trainer(generator, discriminator, dataloader, criterion, optimizer_g, optimizer_d, device)
    trainer.train(epochs=config['epochs'])