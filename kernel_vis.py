import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from cnn import *

# Load the trained model
conv_net = Conv_Net()
conv_net.load_state_dict(torch.load('cnn.pth'))

# Get the weights of the first convolutional layer of the network
first_conv_layer = None
for module in conv_net.modules():
    if isinstance(module, nn.Conv2d):
        first_conv_layer = module
        break

if first_conv_layer is None:
    raise ValueError("No convolutional layer found")

# Get the kernels (weights)
kernels = first_conv_layer.weight.data

# Normalize kernels for visualization
kernels = kernels - kernels.min()
kernels = kernels / kernels.max()

# Grid construction
n_kernels = kernels.size(0)
n_cols = 8
n_rows = (n_kernels + n_cols - 1) // n_cols

# Create a single figure with one set of axes
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

# Calculate the size of each kernel image and padding
kernel_size = 3  # Size of each kernel (assuming 3x3)
padding = 1     # Space between kernels

# Create a large empty array to hold all kernels
full_grid = np.zeros((n_rows * (kernel_size + padding) - padding,
                     n_cols * (kernel_size + padding) - padding))

# Fill the grid with kernels
for i in range(n_kernels):
    row = i // n_cols
    col = i % n_cols
    kernel = kernels[i, 0].numpy()  # Get first channel (grayscale)
    y_start = row * (kernel_size + padding)
    x_start = col * (kernel_size + padding)
    full_grid[y_start:y_start + kernel_size, 
             x_start:x_start + kernel_size] = kernel

# Display the grid
plt.imshow(full_grid, cmap='gray')

# Set axis labels and ticks
plt.xlabel('x')
plt.ylabel('y')

# Set major ticks at multiples of 5
y_ticks = np.arange(0, full_grid.shape[0], 2.5)
x_ticks = np.arange(0, full_grid.shape[1], 5)
plt.yticks(y_ticks)
plt.xticks(x_ticks)

# Add grid
plt.grid(False)

# Add title
plt.title('Figure 1: Kernels learnt at the first conv. layer')

# Adjust layout and save
plt.tight_layout()
plt.savefig('kernel_grid.png', dpi=300, bbox_inches='tight')
plt.close()

# Apply the kernel to the provided sample image.
img = cv2.imread('sample_image.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img / 255.0  # Normalize the image
img = torch.tensor(img).float()
img = img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

# Apply the kernel to the image
with torch.no_grad():
    output = first_conv_layer(img)

# convert output from shape (1, num_channels, output_dim_0, output_dim_1) to (num_channels, 1, output_dim_0, output_dim_1) for plotting.
output = output.squeeze(0)  # Remove batch dimension
output = output.unsqueeze(1)  # Add channel dimension for plotting

# Create a plot that is a grid of images, where each image is the result of applying one kernel to the sample image.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels, the grid might have 4 rows and 8 columns.
# Finally, normalize the values in the grid to be between 0 and 1 before plotting.

# Normalize output for visualization
output = output - output.min()
output = output / output.max()

# Create visualization of transformed images using the same grid layout as before
plt.figure(figsize=(n_cols * 2, n_rows * 2))
for i in range(n_kernels):
    plt.subplot(n_rows, n_cols, i + 1)
    transformed = output[i, 0]  # Get first channel of each kernel's output
    plt.imshow(transformed, cmap='gray')
    plt.title(f'Kernel {i} Output')
    plt.axis('off')

plt.suptitle('Image Transformed by First Layer Kernels')
plt.tight_layout()
plt.savefig('image_transform_grid.png')
plt.close()















