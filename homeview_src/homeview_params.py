"""
This file is based on this repo: https://github.com/chaitanya100100/VAE-for-Image-Generation (models)
"""

n = 4
beta = 5    # Add hyperparameter beta to make it a beta-VAE
img_rows, img_cols, img_chns = int(480/n), int(640/n), 3
latent_dim = 268
intermediate_dim = 1660
epsilon_std = 1.0
epochs = 800
filters = 32
num_conv = 5
batch_size = 128
