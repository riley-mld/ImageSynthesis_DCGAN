# Set data path
dataset_path = "Datasets/celeba"

# Number of workers
num_workers = 2

# Batch size
batch_size = 128

# Size of images
image_size = 64

# Number of colour chanels, 3 for coloured images(i.e. red, green, blue)
num_channels = 3

# Latent vector size
z_size = 100

# Feature maps size in generator
g_feature_size = 64

# Feature maps size in discriminator
d_feature_size = 64

epochs = 10

# Learning rate
lr = 2e-4

# Beta param for optimizers
beta = 0.5



class Configuration():
    """A class to save the hyperparameters and configs."""
    
    def __init__(self):
        """Initialize the class."""
        self.dataset_path = dataset_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.z_size = z_size
        self.g_feature_size = g_feature_size
        self.d_feature_size = d_feature_size
        self.epochs = epochs
        self.lr = lr
        self.beta = beta