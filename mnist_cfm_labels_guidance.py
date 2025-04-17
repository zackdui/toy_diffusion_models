import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchcfm as tcmf
from torchcfm.conditional_flow_matching import * # ExactOptimalTransportConditionalFlowMatcher
from torchdyn.core import NeuralODE
from tqdm import tqdm
import os
from torchvision.utils import save_image
from unet_new.unet import UNetModel
import torch.nn as nn
import torchdiffeq

# Functions

def loadDataset(batch_size = 64):
    mnist_path = "/data/rbg/shared/datasets/MNIST"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist = datasets.MNIST(root=mnist_path, train=True, transform=transform, download=False)
    print(f"Total number of images in training dataset: {len(mnist)}")
    # Drop last will drop the last incomplete batch if it is not divisible by the batch size
    train_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True, drop_last=False)
    
    return train_loader

class TFirstWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # original model expects (x, t)

    def forward(self, t, x):
        return self.model(x, t)

# Actual running of code 
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
batch_size = 64
num_epochs = 1

train_loader = loadDataset(batch_size)
# for i, data in tqdm(enumerate(train_loader)):
#     print(data[0])
#     print(data[1])
#     break

# Sigma is the fixed standard deviation being added to the gaussian noise
sigma = 0.0
# Any model I want to use for the data
# The model should take in times, the data, and an optional condition
# model = tcmf.models.unet.UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1).to(device)
# model = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1).to(device)
model = UNetModel(
    image_size=28,
    in_channels=1,           # grayscale images
    model_channels=32,       # base number of channels
    out_channels=1,          # same as in_channels, for image generation
    channel_mult=(1, 2, 2),
    num_classes = 10,
    num_res_blocks=1,
    attention_resolutions=[1],  # 4x downsampling attention
).to(device)
optimizer = torch.optim.Adam(model.parameters())

# This can sample x_t from p_t(x| x_0, x_1)
# This can compute the conditional flow u_t(x1 | x0)
# The exact allows us to get good pairs of (x0, x1)
# FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
FM = TargetConditionalFlowMatcher(sigma=sigma)

# The first parameter in NeuralODE is the vector field which is the model in our case
# The model needs time t and state x as arguments
# This solver will determine how finely to step through the t's
# node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
model.train()
for epoch in range(num_epochs):
    for i, data in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        # x1 is an example from the final data distribution that we want to generate
        x1 = data[0].to(device)
        y = data[1].to(device)
        # x0 is gaussian noise in the shape of the input to it
        x0 = torch.randn_like(x1)
        # This schedules the t and gives us the xt for that time and the vector field
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        # print("t_shape = ", type(t))
        # print("xt_shape = ", xt.shape)
        drop_label = False
        if torch.rand(1) < 0.1:  # 10% chance
            drop_label = True  # unconditional case
        
        vt = model(t, xt, y, unlabel=drop_label)
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optimizer.step()
    print(f"Loss for epoch {epoch} is {loss}")

torch.save(model.state_dict(), "unet_model_mnist_labels_guidance.pth")

inference_batch = 10
model.eval()
w = 2
with torch.no_grad():
    # The trajectory takes in the x and t_span. Where x is the original data 
    # It is solving the ODE
    # t_span is at what points to return the solution, When it should evaluate the function
    # linspace(start, stop, step)
    # traj.shape = (t_span.shape (int), batch_size, *dim of x)
    # traj = node.trajectory(
    #     torch.randn(inference_batch, 1, 28, 28, device=device),
    #     t_span=torch.linspace(0, 1, 2, device=device), # Evaluate at 0.0 and 1.0
    # )
    traj = torchdiffeq.odeint(
            lambda t, x: (1 + w) * model.forward(t, x, torch.tensor(list(range(10)), device=device)) - w * model.forward(t, x, torch.tensor(list(range(10)), device=device), unlabel=True),
            torch.randn(10, 1, 28, 28, device=device),
            torch.linspace(0, 1, 2, device=device),
            atol=1e-4,
            rtol=1e-4,
            method="dopri5",
        )
# Select all the images at the final timestep
# Then make sure they are the correct size and clip values between [-1, 1]
images = traj[-1, :].view([-1, 1, 28, 28]).clip(-1, 1)
os.makedirs("outputs_mnist_cfm_labels_guidance", exist_ok=True)
for i, img in enumerate(images):
    save_path = f"outputs_mnist_cfm_labels_guidance/sample_{i}.png"
    save_image(img, save_path)

