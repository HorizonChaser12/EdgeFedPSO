import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import copy
import logging
from data_storage import save_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define some constants
num_edge_servers = 10
m = num_edge_servers  # Number of edge servers
k = 100  # Number of clients per edge server
b = 32  # Local batch size
E = 5  # Number of local epochs
learning_rate = 0.01
num_rounds = 10  # Number of communication rounds
num_classes = 10
patience = 3

# Load and preprocess the CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Split the training set into training and validation sets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_size = len(trainset)
val_size = int(0.1 * train_size)
train_size -= val_size
train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=b, shuffle=True, num_workers=5)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=b, shuffle=False, num_workers=5)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Define a simple convolutional neural network (CNN) model with dropout
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)  # Add dropout layer with probability 0.5

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout before the output layer
        x = self.fc2(x)
        return x


# Instantiate the global model
global_model = CNNModel()


def adjust_learning_rate(optimizer, patience, early_stopping_counter):
    if early_stopping_counter >= patience:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1  # Reduce the learning rate by a factor of 0.1


# Placeholder functions for data generation and gradient calculation
def calculate_gradient(model, dataloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001)  # Adding weight decay

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    confusion_matrix = torch.zeros(num_classes, num_classes)

    model.train()  # Set the model to training mode

    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    accuracy = (total_correct / total_samples) * 100.0
    average_loss = total_loss / len(dataloader)

    # Calculate precision, recall, and F1 score
    precision = torch.diag(confusion_matrix) / confusion_matrix.sum(dim=0)
    recall = torch.diag(confusion_matrix) / confusion_matrix.sum(dim=1)
    f1_score = 2 * precision * recall / (precision + recall)

    return average_loss, accuracy, precision, recall, f1_score


# PSOUpdate procedure with PSO (unchanged)
def PSOUpdate(client_id, local_model, dataloader, global_best_model, global_best_score):
    # PSO parameters
    num_particles = 5
    inertia_weight = 0.5
    cognitive_coeff = 2.0
    social_coeff = 2.0

    # Initialize particles with velocities
    particles = [{'model': copy.deepcopy(local_model.state_dict()), 'score': float('inf'),
                  'velocity': {key: torch.zeros_like(value) for key, value in local_model.state_dict().items()}}
                 for _ in range(num_particles)]

    for epoch in range(1, E + 1):
        for particle in particles:
            # Calculate the gradient and update the local model
            average_loss, accuracy, precision, recall, f1_score = calculate_gradient(global_model, dataloader)

            # Update the particle's model if it achieves a better score
            if average_loss < particle['score']:
                particle['model'] = copy.deepcopy(local_model.state_dict())
                particle['score'] = average_loss

        # Update global best model
        best_particle = min(particles, key=lambda x: x['score'])
        if best_particle['score'] < global_best_score:
            global_best_model.load_state_dict(best_particle['model'])
            global_best_score = best_particle['score']

        # Update particles' velocities and models using PSO
        for particle in particles:
            r1, r2 = np.random.rand(), np.random.rand()

            # Perform element-wise subtraction
            cognitive_term = {key: cognitive_coeff * r1 * (particle['model'][key] - local_model.state_dict()[key])
                              for key in local_model.state_dict().keys()}

            social_term = {
                key: social_coeff * r2 * (global_best_model.state_dict()[key] - local_model.state_dict()[key])
                for key in local_model.state_dict().keys()}

            # Perform element-wise addition
            particle['velocity'] = {
                key: inertia_weight * particle['velocity'][key] + cognitive_term[key] + social_term[key]
                for key in local_model.state_dict().keys()}

            # Perform element-wise addition
            particle['model'] = {key: particle['model'][key] + particle['velocity'][key] for key in
                                 local_model.state_dict().keys()}

        # Print the results after each epoch
        logger.info(
            f"Client {client_id} - Epoch {epoch}\tAccuracy: {accuracy:.2f}%\tLoss: {best_particle['score']:.4f}")

        # Update the local model with the best particle's model
        local_model.load_state_dict(best_particle['model'])

    # Return the updated local model and accuracy for logging in GlobalAggregation
    return local_model, accuracy


def calculate_weighted_average(models, weights):
    weighted_params = {}
    total_weight = sum(weights)
    for key in models[0].keys():
        weighted_params[key] = sum(model[key] * weight for model, weight in zip(models, weights)) / total_weight
    return weighted_params


# Modify GlobalAggregation function to integrate validation set usage, regularization, and early stopping
def GlobalAggregation():
    global model_parameters
    optimizer = torch.optim.SGD(global_model.parameters(), lr=0.1, momentum=0.9)

    global_best_model = CNNModel()
    global_best_score = float('inf')

    # Lists to store learning curves
    global_losses = []
    global_accuracies = []

    early_stopping_counter = 0  # Initialize early stopping counter

    for t in range(1, num_rounds + 1):
        weighted_params = []

        for m in range(num_edge_servers):
            # Simulate generating a client's dataloader (replace this with actual client data)
            client_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)

            # Calculate the number of samples per client
            num_samples_per_client = len(client_dataset) // num_edge_servers

            # Create a SubsetRandomSampler to select a subset of the data for each client
            indices = list(range(m * num_samples_per_client, (m + 1) * num_samples_per_client))
            sampler = torch.utils.data.SubsetRandomSampler(indices)

            # Create a dataloader for the client using the SubsetRandomSampler
            client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=b, sampler=sampler)

            # Perform PSOUpdate for each edge server in parallel
            local_model, accuracy = PSOUpdate(m, global_model, client_dataloader, global_best_model, global_best_score)

            # Append the local model parameters for this edge server to the list
            weighted_params.append(local_model.state_dict())

            # Use accuracy for decision-making
            if accuracy < 90:
                adjust_learning_rate(optimizer, patience, early_stopping_counter)  # Adjust learning rate if accuracy is low

        # Calculate the weighted average of model parameters
        weights = [1.0] * num_edge_servers  # Equal weights for simplicity
        weighted_average_params = calculate_weighted_average(weighted_params, weights)
        # Update the global model with the weighted average parameters
        global_model.load_state_dict(weighted_average_params)

        # Calculate the gradient descent and accuracy of the global server on the validation set
        global_loss, global_accuracy, precision, recall, f1_score = calculate_gradient(global_model, valloader)  # Use valloader for evaluation

        # Print recall, precision, and F1 score
        logger.info(f"Round {t}: Recall: {recall}, Precision: {precision}, F1 Score: {f1_score}")

        # Check for early stopping based on the performance on the validation set
        if global_loss < global_best_score:
            global_best_model.load_state_dict(global_model.state_dict())
            global_best_score = global_loss
            # Reset early stopping counter
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                logger.info(f"Early stopping at round {t} due to no improvement on the validation set.")
                break

        # Append results to learning curves
        global_losses.append(global_loss)
        global_accuracies.append(global_accuracy)
        save_data(global_accuracies, '../output_e.pkl')
        save_data(global_losses, '../output_f.pkl')
        # Print the results after each round
        logger.info(f"Round {t}: Global Loss: {global_loss:.4f}, Global Accuracy: {global_accuracy:.2f}%")

        # Use global_accuracy for further decision-making
        if global_accuracy > 95:
            break  # Stop training if global accuracy reaches a certain threshold



def main():
    GlobalAggregation()


# Run the GlobalAggregation procedure
if __name__ == '__main__':
    main()
