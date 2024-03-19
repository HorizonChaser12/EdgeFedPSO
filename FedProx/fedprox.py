import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import copy
import matplotlib.pyplot as plt

# Define some constants
b = 32  # Local batch size
E = 5  # Number of local epochs
learning_rate = 0.001
num_rounds = 3  # Number of communication rounds
num_classes = 10

# Load and preprocess the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=b, shuffle=True)

# Define a simple convolutional neural network (CNN) model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = x.view(-1, 32 * 13 * 13)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the global model
global_model = CNNModel()

# FedProxOptimizer
class FedProxOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.01, mu=0.01):
        defaults = dict(lr=lr, mu=mu)
        super(FedProxOptimizer, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)
                p.data = self.proximal(p.data, group['mu'], group['lr'])

    def proximal(self, params, mu, lr):
        # Flatten the parameters of the global model
        global_params = torch.cat([torch.flatten(v) for v in global_model.state_dict().values()]).to(params.device)

        # Ensure that global_params has the same number of elements as params
        num_elements = params.numel()
        global_params = global_params.repeat(num_elements // global_params.numel() + 1)[:num_elements]

        # Reshape global_params to match the shape of params
        global_params = global_params.view(params.shape)

        # Perform the proximal operation
        return params - mu * lr * (params - global_params)

# GlobalAggregation procedure
def GlobalAggregation():
    # Lists to store learning curves
    global_losses = []
    global_accuracies = []

    optimizer = FedProxOptimizer(global_model.parameters(), lr=learning_rate)

    for t in range(1, num_rounds + 1):
        for epoch in range(1, E + 1):
            for inputs, labels in trainloader:
                optimizer.zero_grad()
                outputs = global_model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                optimizer.step()

            # Print the loss and accuracy after each epoch
            average_loss, accuracy = calculate_loss_and_accuracy(global_model, trainloader)
            print(f"Round {t}, Epoch {epoch}: Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")

            # Append results to learning curves
            global_losses.append(average_loss)
            global_accuracies.append(accuracy)

    # Plot learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_rounds * E + 1), global_losses, marker='o')
    plt.title('Global Loss Over Communication Rounds')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_rounds * E + 1), global_accuracies, marker='o')
    plt.title('Global Accuracy Over Communication Rounds')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

# Function to calculate loss and accuracy
def calculate_loss_and_accuracy(model, dataloader):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = (total_correct / total_samples) * 100.0
    average_loss = total_loss / len(dataloader)

    model.train()

    return average_loss, accuracy

# Run the GlobalAggregation procedure
GlobalAggregation()
