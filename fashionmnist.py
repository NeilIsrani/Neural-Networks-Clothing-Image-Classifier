import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from cnn import *
from ffn import *

'''

In this file you will write end-to-end code to train two neural networks to categorize fashion-mnist data,
one with a feedforward architecture and the other with a convolutional architecture. You will also write code to
evaluate the models and generate plots.

'''


'''

PART 1:
Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
Anything that works is accepted. Please do not change the transforms given below - the autograder assumes these.

'''

transform = transforms.Compose([                            
    transforms.ToImage(),                                  
    transforms.ToDtype(torch.float32, scale=True),        
    transforms.Normalize(mean=[0.5], std=[0.5])           
])

# Choose a reasonable batch size
batch_size = 64


'''

PART 2:
Load the dataset. Make sure to utilize the transform and batch_size from the last section.

'''

# Load training data
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                            download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=0)  # Changed to 0

# Load test data
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                           download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                       shuffle=False, num_workers=0)  # Changed to 0


'''

PART 3:
Complete the model defintion classes in ffn.py and cnn.py. We instantiate the models below.

'''

# Initialize intstances of models
feedforward_net = FF_Net()
conv_net = Conv_Net()



'''

PART 4:
Choose a good loss function and optimizer - you can use the same loss for both networks.

'''

# Choose loss function and optimizers
criterion = nn.CrossEntropyLoss()

# Adam optimizer with learning rate 0.001
optimizer_ffn = optim.Adam(feedforward_net.parameters(), lr=0.001)
optimizer_cnn = optim.Adam(conv_net.parameters(), lr=0.001)



'''

PART 5:
Train both your models, one at a time! (You can train them simultaneously if you have a powerful enough computer,
and are using the same number of epochs, but it is not recommended for this assignment.)

'''

def train_model(model, trainloader, criterion, optimizer, num_epochs, model_name):
    """
    Train a model and track its loss over time
    
    Args:
        model: The neural network model to train
        trainloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        model_name: Name of the model for logging
        
    Returns:
        List of average losses per epoch
    """
    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Calculate average loss for this epoch
        avg_loss = running_loss / len(trainloader)
        losses.append(avg_loss)
        print(f'{model_name} Epoch {epoch + 1}, Loss: {avg_loss:.4f}')
    
    print(f'Finished Training {model_name}')
    return losses

# Training FFN
num_epochs_ffn = 10
ffn_losses = train_model(feedforward_net, trainloader, criterion, optimizer_ffn, 
                        num_epochs_ffn, 'FFN')
torch.save(feedforward_net.state_dict(), 'ffn.pth')

# Training CNN
num_epochs_cnn = 10
cnn_losses = train_model(conv_net, trainloader, criterion, optimizer_cnn, 
                        num_epochs_cnn, 'CNN')
torch.save(conv_net.state_dict(), 'cnn.pth')




'''

PART 6:
Evalute your models! Accuracy should be greater or equal to 80% for both models.

Code to load saved weights commented out below - may be useful for debugging.

'''

# feedforward_net.load_state_dict(torch.load('ffn.pth'))
# conv_net.load_state_dict(torch.load('cnn.pth'))

# Evaluation
correct_ffn = 0
total_ffn = 0
correct_cnn = 0
total_cnn = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        
        # Evaluate FFN
        outputs_ffn = feedforward_net(images)
        _, predicted_ffn = torch.max(outputs_ffn.data, 1)
        total_ffn += labels.size(0)
        correct_ffn += (predicted_ffn == labels).sum().item()
        
        # Evaluate CNN
        outputs_cnn = conv_net(images)
        _, predicted_cnn = torch.max(outputs_cnn.data, 1)
        total_cnn += labels.size(0)
        correct_cnn += (predicted_cnn == labels).sum().item()

print(f'Accuracy of FFN on test images: {100 * correct_ffn / total_ffn}%')
print(f'Accuracy of CNN on test images: {100 * correct_cnn / total_cnn}%')


'''

PART 7:

Check the instructions PDF. You need to generate some plots. 

'''
# Question 1:
#For each neural network, submit a figure containing one image that is classified in-
#correctly by the model, and one image classified correctly by the model. Include clear
#labels that indicate the predicted classes from the model and the true classes the images
#belong to (both human-readable labels, not just the class number). 

# Define Fashion MNIST class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def find_examples(model, testloader, model_name):
    """
    Find one correctly and one incorrectly classified example
    """
    correct_example = None
    incorrect_example = None
    
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Find first correct and incorrect prediction
            for i in range(len(labels)):
                if correct_example is None and predicted[i] == labels[i]:
                    correct_example = (images[i], labels[i], predicted[i])
                if incorrect_example is None and predicted[i] != labels[i]:
                    incorrect_example = (images[i], labels[i], predicted[i])
                if correct_example is not None and incorrect_example is not None:
                    return correct_example, incorrect_example
    return correct_example, incorrect_example

def plot_examples(model, testloader, model_name):
    """
    Plot correct and incorrect examples for a model
    """
    correct, incorrect = find_examples(model, testloader, model_name)
    
    plt.figure(figsize=(10, 5))
    
    # Plot correct example
    plt.subplot(1, 2, 1)
    plt.imshow(correct[0].squeeze(), cmap='gray')
    plt.title(f'Correct Classification\nTrue: {class_names[correct[1]]}\nPredicted: {class_names[correct[2]]}')
    plt.axis('off')
    
    # Plot incorrect example
    plt.subplot(1, 2, 2)
    plt.imshow(incorrect[0].squeeze(), cmap='gray')
    plt.title(f'Incorrect Classification\nTrue: {class_names[incorrect[1]]}\nPredicted: {class_names[incorrect[2]]}')
    plt.axis('off')
    
    plt.suptitle(f'{model_name} Classification Examples')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_examples.png')
    plt.close()

# Plot examples for both models
plot_examples(feedforward_net, testloader, 'FFN')
plot_examples(conv_net, testloader, 'CNN')

# Question 2:
# Plot training losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs_ffn + 1), ffn_losses, label='FFN Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Comparison: FFN')
plt.legend()
plt.grid(True)
plt.savefig('training_losses_ffn.png')
plt.close()

# PLOT 2
plt.figure(figsize=(10, 6))

plt.plot(range(1, num_epochs_cnn + 1), cnn_losses, label='CNN Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Comparison: CNN')
plt.legend()
plt.grid(True)
plt.savefig('training_losses_cnn.png')
plt.close()
'''

YOUR CODE HERE

'''