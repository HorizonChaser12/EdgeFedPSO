import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import copy
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from data_storage import save_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
num_edge_servers = 5
m = num_edge_servers  # Number of edge servers
k = 10  # Number of clients per edge server
b = 32  # Local batch size
E = 5   # Number of local epochs
learning_rate = 0.001
num_rounds = 10  # Number of communication rounds
num_classes = 10

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Define the CNN model
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
global_model = CNNModel().to(device)

# Function to calculate gradients and metrics
def calculate_metrics(model, dataloader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

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

    accuracy = (total_correct / total_samples) * 100.0
    average_loss = total_loss / len(dataloader)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return average_loss, accuracy, precision, recall, f1

# EdgeUpdate function
def EdgeUpdate(client_id, model, dataloader):
    local_model = copy.deepcopy(model)
    local_model.train()
    optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, E + 1):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Log metrics after each epoch
        loss, accuracy, precision, recall, f1 = calculate_metrics(local_model, dataloader)
        logger.info(f"Client {client_id} - Epoch {epoch}: Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

    return local_model.state_dict()

# Function to calculate weighted average of model parameters
def calculate_weighted_average(model_params, weights):
    avg_params = copy.deepcopy(model_params[0])
    for key in avg_params.keys():
        avg_params[key] = sum(w * params[key] for params, w in zip(model_params, weights)) / sum(weights)
    return avg_params

# GlobalAggregation function
def GlobalAggregation():
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

    # Create data loader for the full training set and test set
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=b, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=b, shuffle=False)

    for t in range(1, num_rounds + 1):
        local_models = []
        weights = []

        # EdgeUpdate for each client
        for client_id in range(num_edge_servers):
            local_model_params = EdgeUpdate(client_id, global_model, client_dataloaders[client_id])
            local_models.append(local_model_params)
            weights.append(len(client_dataloaders[client_id].dataset))

        # Aggregate models
        global_model_dict = calculate_weighted_average(local_models, weights)
        global_model.load_state_dict(global_model_dict)

        # Evaluate on training data
        train_loss, train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(global_model, trainloader)

        # Log training metrics
        global_losses.append(train_loss)
        global_accuracies.append(train_accuracy)
        global_precisions.append(train_precision)
        global_recalls.append(train_recall)
        global_f1_scores.append(train_f1)

        # Save metrics
        save_data(global_accuracies, '../Results/EdgeFed_Accuracy.pkl')
        save_data(global_losses, '../Results/EdgeFed_Losses.pkl')
        save_data(global_precisions, '../Results/EdgeFed_Precisions.pkl')
        save_data(global_recalls, '../Results/EdgeFed_Recalls.pkl')
        save_data(global_f1_scores, '../Results/EdgeFed_f1Scores.pkl')

        logger.info(f"Round {t}: Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, "
                    f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-Score: {train_f1:.4f}")

        # Evaluate on test data
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(global_model, testloader)

        logger.info(f"Round {t}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, "
                    f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-Score: {test_f1:.4f}")

# Run the GlobalAggregation procedure
GlobalAggregation()