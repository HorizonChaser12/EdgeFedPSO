import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import copy
import matplotlib.pyplot as plt
from data_storage import save_data
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle  # For saving data in .pkl format

# Define some constants
num_edge_servers = 5
m = num_edge_servers  # Number of edge servers
k = 10  # Number of clients per edge server
b = 32  # Local batch size
E = 5  # Number of local epochs
learning_rate = 0.001
num_rounds = 5  # Number of communication rounds
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


# Placeholder functions for data generation and gradient calculation
def calculate_gradient(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # Disable gradient computation during evaluation
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = (total_correct / total_samples) * 100.0
    average_loss = total_loss / len(dataloader)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    model.train()  # Set the model back to training mode

    return average_loss, accuracy, precision, recall, f1


# EdgeUpdate procedure
def EdgeUpdate(client_id, local_model, dataloader):
    # Create a copy of the local model to avoid modifying it directly
    local_model_copy = copy.deepcopy(local_model)

    # Set the local model's parameters to the global model's parameters
    local_model_copy.load_state_dict(global_model.state_dict())

    # Set the local model to training mode
    local_model_copy.train()

    # Define optimizer and criterion for the local model
    optimizer = optim.SGD(local_model_copy.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, E + 1):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = local_model_copy(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Print the loss and accuracy after each epoch
        average_loss, accuracy, _, _, _ = calculate_gradient(local_model_copy, dataloader)
        print(f"Client {client_id} - Epoch {epoch}: Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Return the state_dict of the locally updated model
    return local_model_copy.state_dict()


# GlobalAggregation procedure
model_parameters = [global_model.state_dict() for _ in range(num_edge_servers)]



def calculate_weighted_average(models, weights):
    weighted_params = {}
    total_weight = sum(weights)
    for key in models[0].keys():
        weighted_params[key] = sum(model[key] * weight for model, weight in zip(models, weights)) / total_weight
    return weighted_params


def GlobalAggregation():
    global model_parameters

    # Lists to store learning curves
    global_losses = []
    global_accuracies = []
    global_precisions = []
    global_recalls = []
    global_f1_scores = []

    for t in range(1, num_rounds + 1):
        weighted_params = []
        for m in range(num_edge_servers):
            client_dataloader = trainloader
            updated_model_dict = EdgeUpdate(m, global_model, client_dataloader)
            model_parameters[m] = updated_model_dict
            weighted_params.append(updated_model_dict)

        weights = [1.0] * num_edge_servers
        weighted_average_params = calculate_weighted_average(weighted_params, weights)

        for key, global_param in global_model.state_dict().items():
            global_param.copy_(weighted_average_params[key])

        global_loss, global_accuracy, global_precision, global_recall, global_f1 = calculate_gradient(global_model, trainloader)

        global_losses.append(global_loss)
        global_accuracies.append(global_accuracy)
        global_precisions.append(global_precision)
        global_recalls.append(global_recall)
        global_f1_scores.append(global_f1)

        # Save metrics
        save_data(global_accuracies, '../output_accuracies.pkl')
        save_data(global_losses, '../output_losses.pkl')
        save_data(global_precisions, '../output_precisions.pkl')
        save_data(global_recalls, '../output_recalls.pkl')
        save_data(global_f1_scores, '../output_f1_scores.pkl')

        print(
            f"Round {t}: Global Loss: {global_loss:.4f}, Global Accuracy: {global_accuracy:.2f}%, Precision: {global_precision:.2f}, Recall: {global_recall:.2f}, F1 Score: {global_f1:.2f}")


# Run the GlobalAggregation procedure
GlobalAggregation()
