import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import copy
import logging
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from Research.Plotting.data_storage import save_data
import numpy as np
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
num_edge_servers = 10
num_clients = 100  # Improvement 5: Adjust to 100 mobile devices
k = num_clients // num_edge_servers  # Clients per edge server
b = 16  # Local batch size
E = 5  # Number of local epochs
initial_learning_rate = 0.001  # Reduced initial learning rate
num_rounds = 50  # Number of communication rounds
num_classes = 10

# Improvement 2: Add bandwidth simulation
bandwidth_client_edge = 8 * 1024 * 1024  # 8 MB/s (in bits/s)
bandwidth_edge_cloud = 3 * 1024 * 1024  # 3 MB/s (in bits/s)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)


# Define the split CNN model with dropout
class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        return x


class EdgeModel(nn.Module):
    def __init__(self):
        super(EdgeModel, self).__init__()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# Function to create non-IID data distribution
def create_non_iid_data(dataset, num_edge_servers, k, alpha=0.5):
    num_samples = len(dataset)
    total_clients = num_edge_servers * k
    client_data = [[] for _ in range(total_clients)]

    # Sort the indices by label
    indices_by_label = [[] for _ in range(num_classes)]
    for idx, target in enumerate(dataset.targets):
        label = target.item()
        indices_by_label[label].append(idx)

    # Use Dirichlet distribution to allocate samples
    for indices in indices_by_label:
        proportions = np.random.dirichlet(np.repeat(alpha, total_clients))
        # Ensure we don't assign more indices than available
        num_indices = len(indices)
        if num_indices > 0:
            split_points = (np.cumsum(proportions) * num_indices).astype(int)
            split_indices = np.split(indices, split_points[:-1])
            for client_id, client_indices in enumerate(split_indices):
                client_data[client_id].extend(client_indices)

    # Ensure all clients have at least one sample
    for client_id, data in enumerate(client_data):
        if len(data) == 0:
            # If a client has no data, give it a random sample
            random_label = np.random.randint(num_classes)
            if indices_by_label[random_label]:
                random_index = np.random.choice(indices_by_label[random_label])
                client_data[client_id].append(random_index)
                indices_by_label[random_label].remove(random_index)

    return [torch.utils.data.Subset(dataset, indices) for indices in client_data]


# Usage remains the same
client_datasets = create_non_iid_data(trainset, num_edge_servers, k)


# Improvement 2: Add function to simulate data transfer time
def simulate_transfer_time(data_size, bandwidth, variability=0.2):
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive")
    base_time = data_size / bandwidth
    return base_time * (1 + random.uniform(-variability, variability))



def get_bandwidth_scenario(scenario):
    if scenario == "best":
        return 10 * 1024 * 1024, 5 * 1024 * 1024  # 10 MB/s local, 5 MB/s global
    elif scenario == "worst":
        return 6 * 1024 * 1024, 1 * 1024 * 1024  # 6 MB/s local, 1 MB/s global
    else:  # default
        return 8 * 1024 * 1024, 3 * 1024 * 1024  # 8 MB/s local, 3 MB/s global


# Usage
bandwidth_client_edge, bandwidth_edge_cloud = get_bandwidth_scenario("default")


def calculate_communication_cost(local_transfer_time, edge_to_cloud_transfer_time):
    total_cost = local_transfer_time + edge_to_cloud_transfer_time
    logger.info(f"Local transfer time: {local_transfer_time:.2f}, Edge-to-cloud transfer time: {edge_to_cloud_transfer_time:.2f}")
    return total_cost


# Improvement 4: Add learning rate decay
def adjust_learning_rate(optimizer, round):
    lr = initial_learning_rate * (0.99 ** round)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Function to calculate metrics
def calculate_metrics(client_model, edge_model, dataloader):
    client_model.eval()
    edge_model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            client_output = client_model(inputs)
            outputs = edge_model(client_output)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    return avg_loss, accuracy, precision, recall, f1


# EdgeUpdate function
def EdgeUpdate(client_id, client_model, edge_model, dataloader, round):
    client_model.train()
    edge_model.train()
    optimizer = optim.SGD(list(client_model.parameters()) + list(edge_model.parameters()),
                          lr=initial_learning_rate, momentum=0.9, weight_decay=1e-4)

    # Improvement 4: Apply learning rate decay
    adjust_learning_rate(optimizer, round)

    criterion = nn.CrossEntropyLoss()
    total_client_output_size = 0
    for epoch in range(E):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Simulate client computation and data transfer
            client_output = client_model(inputs)
            client_output_size = client_output.nelement() * client_output.element_size()
            total_client_output_size += client_output_size
            transfer_time = simulate_transfer_time(client_output_size, bandwidth_client_edge)

            # Simulate edge server computation
            outputs = edge_model(client_output)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        logger.info(f"Round {round} - Client {client_id} - Epoch {epoch}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    return client_model.state_dict(), edge_model.state_dict(), transfer_time, total_client_output_size


# Function to calculate weighted average of model parameters
def calculate_weighted_average(model_params, weights):
    avg_params = copy.deepcopy(model_params[0])
    for key in avg_params.keys():
        avg_params[key] = sum(w * params[key] for params, w in zip(model_params, weights)) / sum(weights)
    return avg_params


def GlobalAggregation():
    global_client_model = ClientModel().to(device)
    global_edge_model = EdgeModel().to(device)

    communication_costs = []
    global_losses = []
    global_accuracies = []
    global_precisions = []
    global_recalls = []
    global_f1_scores = []

    client_dataloaders = [torch.utils.data.DataLoader(dataset, batch_size=b, shuffle=True) for dataset in client_datasets]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

    total_comm_cost = 0

    for t in range(1, num_rounds + 1):
        client_models = []
        edge_models = []
        weights = []
        total_local_transfer_time = 0
        total_client_output_size = 0

        for client_id in range(num_clients):
            client_model = copy.deepcopy(global_client_model)
            edge_model = copy.deepcopy(global_edge_model)

            client_model_params, edge_model_params, transfer_time, client_output_size = EdgeUpdate(
                client_id, client_model, edge_model, client_dataloaders[client_id], t)

            client_models.append(client_model_params)
            edge_models.append(edge_model_params)
            weights.append(len(client_dataloaders[client_id].dataset))
            total_local_transfer_time += transfer_time
            total_client_output_size += client_output_size

        # Simulate edge to cloud transfer time
        edge_to_cloud_size = sum(p.nelement() * p.element_size() for p in global_client_model.parameters()) + \
                             sum(p.nelement() * p.element_size() for p in global_edge_model.parameters())
        edge_to_cloud_time = simulate_transfer_time(edge_to_cloud_size, bandwidth_edge_cloud)

        # Calculate round communication cost
        round_comm_cost = calculate_communication_cost(
            local_transfer_time=total_local_transfer_time,
            edge_to_cloud_transfer_time=edge_to_cloud_time
        )
        total_comm_cost += round_comm_cost

        logger.info(f"Round {t} - Communication cost: {round_comm_cost:.2f} seconds")
        logger.info(f"Round {t} - Client output size: {total_client_output_size} bits")
        logger.info(f"Round {t} - Edge-to-cloud size: {edge_to_cloud_size} bits")

        global_client_dict = calculate_weighted_average(client_models, weights)
        global_edge_dict = calculate_weighted_average(edge_models, weights)

        global_client_model.load_state_dict(global_client_dict)
        global_edge_model.load_state_dict(global_edge_dict)

        train_loss, train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(
            global_client_model, global_edge_model, trainloader)

        communication_costs.append(round_comm_cost)
        global_losses.append(train_loss)
        global_accuracies.append(train_accuracy)
        global_precisions.append(train_precision)
        global_recalls.append(train_recall)
        global_f1_scores.append(train_f1)

        save_data(global_accuracies, '../Results/EdgeFed_Accuracyb16e5.pkl')
        save_data(global_losses, '../Results/EdgeFed_Lossesb16e5.pkl')
        save_data(global_precisions, '../Results/EdgeFed_Precisionsb16e5.pkl')
        save_data(global_recalls, '../Results/EdgeFed_Recallsb16e5.pkl')
        save_data(global_f1_scores, '../Results/EdgeFed_f1Scoresb16e5.pkl')
        save_data(communication_costs, '../Results/EdgeFed_communicationCostb16e5.pkl')

        logger.info(f"Round {t}: Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, "
                    f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-Score: {train_f1:.4f}")

        test_loss, test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(
            global_client_model, global_edge_model, testloader)

        logger.info(f"Round {t}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, "
                    f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-Score: {test_f1:.4f}")

    # After the training loop
    average_comm_cost = total_comm_cost / num_rounds
    logger.info(f"Average communication cost per round: {average_comm_cost:.2f} seconds")

    comm_cost_data = {
        'total_comm_cost': total_comm_cost,
        'average_comm_cost': average_comm_cost,
        'num_rounds': num_rounds
    }
    save_data(comm_cost_data, '../Results/EdgeFed_communicationCostb16e5.pkl')


# Run the GlobalAggregation procedure
GlobalAggregation()
