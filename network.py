import torch
import torch.nn as nn

from config import Configuration

config = Configuration()

# Weights initilization on the networks based on DCGAN paper
def weights_init(n):
    classname = n.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(n.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(n.weight.data, 1.0, 0.02)
        nn.init.constant_(n.bias.data, 0)

        
class Generator(nn.Module):
    """A class to define generator network."""
    def __init__(self):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
                                    # Input Z is going through a transposed convolution
                                    nn.ConvTranspose2d(config.z_size, config.g_feature_size * 8, 4, 1, 0, bias=False),
                                    nn.BatchNorm2d(config.g_feature_size * 8),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    # Size: (g_feature_size * 8) * 4 * 4
                                    nn.ConvTranspose2d(config.g_feature_size * 8, config.g_feature_size * 4, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(config.g_feature_size * 4),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    # Size: (g_feature_size * 4) * 8 * 8
                                    nn.ConvTranspose2d(config.g_feature_size * 4, config.g_feature_size * 2, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(config.g_feature_size * 2),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    # Size: (g_feature_size * 2) * 16 * 16
                                    nn.ConvTranspose2d(config.g_feature_size * 2, config.g_feature_size, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(config.g_feature_size),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    # Size: g_feature_size * 32 * 32
                                    nn.ConvTranspose2d(config.g_feature_size, config.num_channels, 4, 2, 1, bias=False),
                                    nn.Tanh()
                                    # Size: num_channels * 64 * 64
                                    )
    def forward(self, input):
        """Forward Pass."""
        return self.network(input)
    
    
class Discriminator(nn.Module):
    """A class to define discriminator network."""
    def __init__(self):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(

        
                                     # Input size: num_channels * 64 * 64
                                     nn.Conv2d(config.num_channels, config.d_feature_size, 4, 2, 1, bias=False),
                                     nn.LeakyReLU(0.2,  inplace=True),
                                     # Size: d_feature_size * 32 * 32
                                     nn.Conv2d(config.d_feature_size, config.d_feature_size * 2, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(config.d_feature_size * 2),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     # Size: (d_feature_size * 2) * 16 * 16
                                     nn.Conv2d(config.d_feature_size *2, config.d_feature_size * 4, 4, 2, 1, bias= False),
                                     nn.BatchNorm2d(config.d_feature_size * 4),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     # Size: (d_feature_size * 4) * 8 * 8
                                     nn.Conv2d(config.d_feature_size * 4, config.d_feature_size * 8, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(config.d_feature_size * 8),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     # Size: (d_feature_size * 8) * 4 * 4
                                     nn.Conv2d(config.d_feature_size * 8, 1, 4, 1, 0, bias=False),
                                     nn.Sigmoid()
                                     )
    def forward(self, input):
        """Forward pass."""
        return self.network(input)