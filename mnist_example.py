import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from accelerate import Accelerator
from itertools import pairwise
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from typing import Optional
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import matplotlib
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from smalldiffusion import (
    ScheduleLogLinear, training_loop, samples, Swissroll, TimeInputMLP, Schedule,
    ModelMixin, get_sigma_embeds
)
import os

def loadDataset():
    mnist_path = "/data/rbg/shared/datasets/MNIST"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist = torchvision.datasets.MNIST(root=mnist_path, train=True, transform=transform, download=False)
    train_loader = DataLoader(mnist, batch_size=64, shuffle=True)

    # Get the first image and its label from the dataset
    first_image, first_label = mnist[0]  # Get first sample

    # Print tensor properties
    print(f"Tensor Shape: {first_image.shape}")  # (C, H, W) for PyTorch (should be (1, 28, 28))
    print(f"Tensor Type: {first_image.dtype}")   # Should be torch.float32
    print(f"Min Pixel Value: {first_image.min().item()}")  # Minimum pixel value
    print(f"Max Pixel Value: {first_image.max().item()}")  # Maximum pixel value
    print(f"Label: {first_label}")  # Corresponding digit (0-9)

    # import matplotlib.pyplot as plt

    # # Convert tensor to numpy and plot
    # plt.imshow(first_image.squeeze(), cmap="gray")  # Remove channel dimension (1, 28, 28) -> (28, 28)
    # plt.title(f"Label: {first_label}")
    # plt.axis("off")  # Hide axes
    # plt.show()
    # plt.savefig("mnist_sample.png")
    return train_loader

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid() # Make outputs between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

def model():
    return UNet()
    # Define a simple U-Net-like model for denoising
# class SimpleDenoiser(nn.Module):
#     def __init__(self):
#         super(SimpleDenoiser, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 512)
#         self.fc2 = nn.Linear(512, 512)
#         self.fc3 = nn.Linear(512, 28 * 28)
    
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#     pass

def sigmaSchedule():
    # Define Loglinear noise schedule
    T = 100  # Number of timesteps
    beta_start, beta_end = 0.0001, 0.02
    beta = torch.exp(torch.linspace(np.log(beta_start), np.log(beta_end), T))
    # schedule = ScheduleLogLinear(N=500, sigma_min=0.005, sigma_max=10)
    # sx, sy = get_sigma_embeds(len(schedule), schedule.sigmas).T
    return beta

def forward_diffusion(image, time, noise, beta):
    # alpha helps the idffusion
    alpha_t = (1 - beta[time]).sqrt()
    sigma_t = beta[time].sqrt()
    while len(sigma_t.shape) < len(x.shape):
        sigma_t = sigma_t.unsqueeze(-1)
    return alpha_t * x + sigma_t * noise

def training(num_epochs, model, dataloader, lr = 1e-3):
    accelerator = Accelerator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    beta = sigmaSchedule()
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for images, _ in dataloader:
            images = images.to(device)
            images = images.view(-1, 1, 28, 28)  # Ensure correct shape for CNN
            
            t = torch.randint(0, T, (images.size(0),), device=device)
            noise = torch.randn_like(images).to(device)
            noisy_images = forward_diffusion(images, t, noise, beta)
            
            predicted_noise = model(noisy_images)
            loss = F.mse_loss(predicted_noise, noise)
            yield loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        pass
def run(num_epochs = 1):
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    dataloader = DataLoader()
    losses = list(training(num_epochs, model, dataloader, lr = 1e-3))
    plt.plot(moving_average(losses, 100))
    
def inference():
    # Generate images by denoising
    beta = sigmaSchedule()
    @torch.no_grad()
    def generate_images(num_samples=10):
        x = torch.randn((num_samples, 1, 28, 28)).to(device)
        for t in reversed(range(T)):
            predicted_noise = model(x)
            alpha_t = (1 - beta[t]).sqrt()
            sigma_t = beta[t].sqrt()
            while len(sigma_t.shape) < len(x.shape):
                sigma_t = sigma_t.unsqueeze(-1)
            x = (x - sigma_t * predicted_noise) / torch.clamp(alpha_t, min=1e-6)
        return x.cpu().view(-1, 28, 28)

    # Display generated images
    generated_images = generate_images()
    save_dir = "generated_images"
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(generated_images):
        plt.imsave(f"{save_dir}/image_{i}.png", img.numpy(), cmap="gray")
    # fig, axes = plt.subplots(1, len(generated_images), figsize=(10, 2))
    # for img, ax in zip(generated_images, axes):
    #     ax.imshow(img, cmap="gray")
    #     ax.axis("off")
    # plt.show()