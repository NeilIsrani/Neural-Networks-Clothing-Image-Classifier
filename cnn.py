import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''

In this file you will write the model definition for a convolutional neural network. 

Please only complete the model definition and do not include any training code.

The model should be a convolutional neural network, that accepts 28x28 grayscale images as input, and outputs a tensor of size 10.
The number of layers/kernels, kernel sizes and strides are up to you. 

Please refer to the following for more information about convolutions, pooling, and convolutional layers in PyTorch:

    - https://deeplizard.com/learn/video/YRhxdVk_sIs
    - https://deeplizard.com/resource/pavq7noze2
    - https://deeplizard.com/resource/pavq7noze3
    - https://setosa.io/ev/image-kernels/
    - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html


Whether you need to normalize outputs using softmax depends on your choice of loss function. PyTorch documentation is available at
https://pytorch.org/docs/stable/index.html, and will specify whether a given loss funciton requires normalized outputs or not.

'''

class Conv_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, 3)  # 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, 3)  # 32 input channels, 64 output channels, 3x3 kernel
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 5 * 5, 512)  # Size calculated based on input transformations
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)  # 10 output classes
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Apply first conv layer and pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply second conv layer and pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 5 * 5)
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
        
