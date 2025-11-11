import torch
import torch.nn as nn

class AdaptiveInstanceLayerNorm(nn.Module):
    def __init__(self, num_features, style_dim):
        super(AdaptiveInstanceLayerNorm, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)
        self.style_scale = nn.Linear(style_dim, num_features)
        self.style_bias = nn.Linear(style_dim, num_features)

    def forward(self, x, style):
        normalized = self.instance_norm(x)
        scale = self.style_scale(style).unsqueeze(2).unsqueeze(3)
        bias = self.style_bias(style).unsqueeze(2).unsqueeze(3)
        return normalized * (1 + scale) + bias

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.instancenorm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.instancenorm2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        skip = x
        x = self.conv1(x)
        x = self.instancenorm1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.instancenorm2(x)
        return x + skip

class ResnetGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, ngf):
        super(ResnetGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_dim, ngf, kernel_size=7, padding=3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            *[ResnetBlock(ngf * 2, ngf * 2) for _ in range(8)],
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, output_dim, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, ndf):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_dim, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

class UGATIT(nn.Module):
    def __init__(self, style_dim):
        super(UGATIT, self).__init__()
        self.genA2B = ResnetGenerator(3, 3, 64)
        self.genB2A = ResnetGenerator(3, 3, 64)
        self.disGA = Discriminator(3, 64)
        self.disGB = Discriminator(3, 64)
        self.disLA = Discriminator(3, 64)
        self.disLB = Discriminator(3, 64)
        self.style_dim = style_dim

    def to(self, device):
        super().to(device)
        for module in [self.genA2B, self.genB2A, self.disGA, self.disGB, self.disLA, self.disLB]:
            module.to(device)

    def parameters(self):
        return list(self.genA2B.parameters()) + list(self.genB2A.parameters()) + \
               list(self.disGA.parameters()) + list(self.disGB.parameters()) + \
               list(self.disLA.parameters()) + list(self.disLB.parameters())