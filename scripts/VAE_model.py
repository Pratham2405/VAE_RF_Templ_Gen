import torch
import torch.nn as nn
# import torch.nn.functional as F

# Basic Residual Block for Conv2d (used in both encoder and decoder)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, upsample=False, transpose=False):
        super().__init__()
        stride = 2 if downsample or upsample else 1
        
        # Choose convolution type: standard or transposed for decoder
        if transpose:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=stride, padding=1, output_padding=(stride-1 if stride > 1 else 0))
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
            
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        if transpose:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1)
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsample or upsample shortcut if needed
        self.shortcut = None
        if downsample and (in_channels != out_channels or stride > 1):
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        elif upsample and (in_channels != out_channels or stride > 1):
            self.shortcut = nn.ConvTranspose2d(in_channels, out_channels, 1, stride=stride, output_padding=(stride-1 if stride > 1 else 0))
        elif in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out += identity
        out = self.relu(out)
        return out

class VanillaVAE(nn.Module):
    def __init__(self, feature_dim, latent_dim, input_size, input_channels=4):
        super().__init__()
        # Encoder: 3 ResNet blocks
        self.enc_block1 = ResidualBlock(input_channels, feature_dim, downsample=True)
        self.enc_block2 = ResidualBlock(feature_dim, feature_dim, downsample=True)
        self.enc_block3 = ResidualBlock(feature_dim, feature_dim, downsample=True)

        # Compute conv_shape after 3 downsamples with stride=2
        conv_shape = input_size // 8  # Integer division; adjust as needed
        flat_dim = feature_dim * conv_shape * conv_shape

        # Latent space projection
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # Decoder: fc then reshape
        self.fc_decode = nn.Linear(latent_dim, flat_dim)

        # Decoder: upsampling + 3 ResNet blocks with transposed convs
        self.dec_block1 = ResidualBlock(feature_dim, feature_dim, upsample=True, transpose=True)
        self.dec_block2 = ResidualBlock(feature_dim, feature_dim, upsample=True, transpose=True)
        self.dec_block3 = ResidualBlock(feature_dim, input_channels, upsample=True, transpose=True)

        # Output activation: sigmoid (if features normalized to [0,1])
        # self.out_act = nn.Sigmoid()

    def encode(self, x):
        x = self.enc_block1(x)
        x = self.enc_block2(x)
        x = self.enc_block3(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        batch_size = z.size(0)
        # Reshape to feature map for decoder
        feature_dim = self.dec_block1.conv1.in_channels
        conv_shape = int((x.size(1) // feature_dim) ** 0.5)
        x = x.view(batch_size, feature_dim, conv_shape, conv_shape)
        x = self.dec_block1(x)
        x = self.dec_block2(x)
        x = self.dec_block3(x)
        # x = self.out_act(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
