import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import copy
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from Plotting.data_storage import save_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
num_edge_servers = 5
m = num_edge_servers  # Number of edge servers
k = 10  # Number of clients per edge server
b = 64  # Local batch size
E = 5  # Number of local epochs
learning_rate = 0.01
num_rounds = 10  # Number of communication rounds
num_classes = 10

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess the CIFAR-10 dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


# Define the CNN model for CIFAR-10
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Instantiate the global model
global_model = CNNModel().to(device)


# Function to calculate gradients and metrics
def calculate_gradient(model, dataloader):
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Calculate gradients
        loss.backward()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    accuracy = (total_correct / total_samples) * 100.0
    average_loss = total_loss / len(dataloader)

    return average_loss, accuracy, all_labels, all_predictions


# Function to calculate metrics
def calculate_metrics(true_labels, predictions):
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    return precision, recall, f1


# Modified LocalUpdate function (without PSO)
def LocalUpdate(client_id, global_model, dataloader):
    local_model = copy.deepcopy(global_model)
    local_model.train()
    optimizer = optim.SGD(local_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    for epoch in range(1, E + 1):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        total_samples = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            epoch_accuracy += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss /= total_samples
        epoch_accuracy = (epoch_accuracy / total_samples) * 100.0
        logger.info(f"Client {client_id} - Epoch {epoch}\tAccuracy: {epoch_accuracy:.2f}%\tLoss: {epoch_loss:.4f}")

    return local_model.state_dict()


# Function to adjust learning rate
def adjust_learning_rate(optimizer, decay_rate=0.95):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate


# Function to calculate weighted average of model parameters
def calculate_weighted_average(weighted_params, weights):
    avg_params = copy.deepcopy(weighted_params[0])
    total_weight = sum(weights)
    for key in avg_params.keys():
        avg_params[key] = sum(weight * param[key] for weight, param in zip(weights, weighted_params)) / total_weight
    return avg_params


# Load test dataset
testloader = torch.utils.data.DataLoader(testset, batch_size=b, shuffle=False)


def evaluate_global_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    average_loss = total_loss / len(dataloader)
    accuracy = (total_correct / total_samples) * 100.0
    precision, recall, f1 = calculate_metrics(all_labels, all_predictions)
    return average_loss, accuracy, precision, recall, f1


# Modified GlobalAggregation function (EdgeFed without PSO)
def GlobalAggregation():
    global_model = CNNModel().to(device)

    global_losses = []
    global_accuracies = []
    global_precisions = []
    global_recalls = []
    global_f1_scores = []

    # Create data loaders for clients
    client_dataloaders = []
    for i in range(num_edge_servers):
        indices = list(range(i, len(trainset), num_edge_servers))
        subset = torch.utils.data.Subset(trainset, indices)
        client_dataloaders.append(torch.utils.data.DataLoader(subset, batch_size=b, shuffle=True))

    # Create data loader for the full training set
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=b, shuffle=True)

    # Create test data loader
    testloader = torch.utils.data.DataLoader(testset, batch_size=b, shuffle=False)

    optimizer = optim.SGD(global_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    for t in range(1, num_rounds + 1):
        local_weights = []

        # Local update for each client
        for client_id in range(num_edge_servers):
            local_model = LocalUpdate(client_id, global_model, client_dataloaders[client_id])
            local_weights.append(local_model)

        # Aggregate weights
        weights = [1.0] * num_edge_servers
        global_weights = calculate_weighted_average(local_weights, weights)
        global_model.load_state_dict(global_weights)

        # Evaluate on training data
        global_loss, global_accuracy, true_labels, predictions = calculate_gradient(global_model, trainloader)
        precision, recall, f1 = calculate_metrics(true_labels, predictions)

        # Log training metrics
        global_losses.append(global_loss)
        global_accuracies.append(global_accuracy)
        global_precisions.append(precision)
        global_recalls.append(recall)
        global_f1_scores.append(f1)

        save_data(global_accuracies, '../Results/EdgeFed_CIFAR10_Accuracy.pkl')
        save_data(global_losses, '../Results/EdgeFed_CIFAR10_Losses.pkl')
        save_data(global_precisions, '../Results/EdgeFed_CIFAR10_Precisions.pkl')
        save_data(global_recalls, '../Results/EdgeFed_CIFAR10_Recalls.pkl')
        save_data(global_f1_scores, '../Results/EdgeFed_CIFAR10_f1Scores.pkl')

        logger.info(f"Round {t}: Training Loss: {global_loss:.4f}, Training Accuracy: {global_accuracy:.2f}%, "
                    f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

        # Evaluate on test data
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_global_model(global_model, testloader)

        logger.info(f"Round {t}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, "
                    f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-Score: {test_f1:.4f}")

        # Adjust learning rate
        adjust_learning_rate(optimizer)


if __name__ == "__main__":
    GlobalAggregation()
