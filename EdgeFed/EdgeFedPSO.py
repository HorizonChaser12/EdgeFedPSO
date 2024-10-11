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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define constants
NUM_EDGE_SERVERS = 10
NUM_CLIENTS = 100
K = NUM_CLIENTS // NUM_EDGE_SERVERS  # Clients per edge server
BATCH_SIZE = 16
LOCAL_EPOCHS = 10
INITIAL_LEARNING_RATE = 0.001
NUM_ROUNDS = 5
NUM_CLASSES = 10

# Bandwidth simulation
LOCAL_BANDWIDTHS = [6 * 1024 * 1024, 8 * 1024 * 1024, 10 * 1024 * 1024]  # Local: 6 MB/s, 8 MB/s, 10 MB/s
GLOBAL_BANDWIDTHS = [1 * 1024 * 1024, 3 * 1024 * 1024, 5 * 1024 * 1024]  # Global: 1 MB/s, 3 MB/s, 5 MB/s

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
        self.fc2 = nn.Linear(128, NUM_CLASSES)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class Particle:
    def __init__(self, model):
        self.device = next(model.parameters()).device
        self.position = {name: param.clone().detach().to(self.device) for name, param in model.named_parameters()}
        self.velocity = {name: torch.zeros_like(param).to(self.device) for name, param in model.named_parameters()}
        self.best_position = {name: param.clone().detach().to(self.device) for name, param in model.named_parameters()}
        self.best_fitness = float('-inf')

    def update_velocity(self, global_best_position, w=0.5, c1=1.5, c2=1.5):
        if global_best_position is None:
            return

        for name in self.position:
            r1, r2 = torch.rand(1).to(self.device), torch.rand(1).to(self.device)
            cognitive_component = c1 * r1 * (self.best_position[name] - self.position[name])
            social_component = c2 * r2 * (global_best_position[name].to(self.device) - self.position[name])
            self.velocity[name] = w * self.velocity[name] + cognitive_component + social_component

    def update_position(self):
        for name in self.position:
            self.position[name] += self.velocity[name]

    def update_best(self, fitness):
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = {name: param.clone().detach() for name, param in self.position.items()}

    def to(self, device):
        self.device = device
        self.position = {name: param.to(device) for name, param in self.position.items()}
        self.velocity = {name: param.to(device) for name, param in self.velocity.items()}
        self.best_position = {name: param.to(device) for name, param in self.best_position.items()}
        return self


def create_non_iid_data(dataset, num_edge_servers, k, alpha=0.5):
    total_clients = num_edge_servers * k
    client_data = [[] for _ in range(total_clients)]

    indices_by_label = [[] for _ in range(NUM_CLASSES)]
    for idx, target in enumerate(dataset.targets):
        label = target.item()
        indices_by_label[label].append(idx)

    for indices in indices_by_label:
        proportions = np.random.dirichlet(np.repeat(alpha, total_clients))
        num_indices = len(indices)
        if num_indices > 0:
            split_points = (np.cumsum(proportions) * num_indices).astype(int)
            split_indices = np.split(indices, split_points[:-1])
            for client_id, client_indices in enumerate(split_indices):
                client_data[client_id].extend(client_indices)

    # Ensure all clients have at least one sample
    for client_id, data in enumerate(client_data):
        if len(data) == 0:
            random_label = np.random.randint(NUM_CLASSES)
            if indices_by_label[random_label]:
                random_index = np.random.choice(indices_by_label[random_label])
                client_data[client_id].append(random_index)
                indices_by_label[random_label].remove(random_index)

    return [torch.utils.data.Subset(dataset, indices) for indices in client_data]


client_datasets = create_non_iid_data(trainset, NUM_EDGE_SERVERS, K)


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


def calculate_communication_cost(local_transfer_time, edge_to_cloud_transfer_time):
    total_cost = local_transfer_time + edge_to_cloud_transfer_time
    logger.info(
        f"Local transfer time: {local_transfer_time:.2f}, Edge-to-cloud transfer time: {edge_to_cloud_transfer_time:.2f}"
    )
    return total_cost


def adjust_learning_rate(optimizer, round):
    lr = INITIAL_LEARNING_RATE * (0.99 ** round)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

            all_preds.append(predicted)
            all_targets.append(targets)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)

    # Calculate precision, recall, and F1 score using PyTorch operations
    num_classes = NUM_CLASSES  # Assuming NUM_CLASSES is defined globally
    true_positives = torch.zeros(num_classes, device=device)
    predicted_positives = torch.zeros(num_classes, device=device)
    actual_positives = torch.zeros(num_classes, device=device)

    for c in range(num_classes):
        true_positives[c] = ((all_preds == c) & (all_targets == c)).sum().float()
        predicted_positives[c] = (all_preds == c).sum().float()
        actual_positives[c] = (all_targets == c).sum().float()

    precision = true_positives / (predicted_positives + 1e-7)  # Add small epsilon to avoid division by zero
    recall = true_positives / (actual_positives + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    # Calculate weighted averages
    weights = actual_positives / actual_positives.sum()
    precision_weighted = (precision * weights).sum().item()
    recall_weighted = (recall * weights).sum().item()
    f1_weighted = (f1 * weights).sum().item()

    return avg_loss, accuracy, precision_weighted, recall_weighted, f1_weighted


def EdgeUpdatePSO(client_id, particle, edge_model, dataloader, round, global_best_position, bandwidth_client_edge):
    device = next(edge_model.parameters()).device
    particle = particle.to(device)

    particle.update_velocity(global_best_position)
    particle.update_position()

    client_model = ClientModel().to(device)
    client_model.load_state_dict(particle.position)

    client_model.train()
    edge_model.train()
    optimizer = optim.SGD(list(client_model.parameters()) + list(edge_model.parameters()),
                          lr=INITIAL_LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

    adjust_learning_rate(optimizer, round)

    criterion = nn.CrossEntropyLoss()
    total_client_output_size = 0

    for epoch in range(LOCAL_EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            client_output = client_model(inputs)
            client_output_size = client_output.nelement() * client_output.element_size()
            total_client_output_size += client_output_size
            transfer_time = simulate_transfer_time(client_output_size, bandwidth_client_edge)

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
        logger.info(
            f"Round {round} - Client {client_id} - Epoch {epoch}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%"
        )

    fitness = epoch_acc  # Using accuracy as fitness
    particle.update_best(fitness)

    return particle, edge_model.state_dict(), transfer_time, total_client_output_size, fitness


def calculate_weighted_average(model_params, weights):
    avg_params = copy.deepcopy(model_params[0])
    for key in avg_params.keys():
        avg_params[key] = sum(w * params[key] for params, w in zip(model_params, weights)) / sum(weights)
    return avg_params


def GlobalAggregationPSO():
    results = []  # Store results for different bandwidth scenarios

    for local_bw in LOCAL_BANDWIDTHS:
        for global_bw in GLOBAL_BANDWIDTHS:
            global_edge_model = EdgeModel().to(device)
            particles = [Particle(ClientModel().to(device)) for _ in range(NUM_CLIENTS)]
            global_best_position = None
            global_best_fitness = float('-inf')

            global_losses = []
            global_accuracies = []
            global_precisions = []
            global_recalls = []
            global_f1_scores = []

            client_dataloaders = [torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) for dataset
                                  in client_datasets]
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=False)
            testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

            total_round_comm_costs = 0
            total_client_output_size = 0
            total_edge_to_cloud_size = 0

            for t in range(1, NUM_ROUNDS + 1):
                round_transfer_time = 0
                round_client_output_size = 0
                edge_models = []
                fitnesses = []

                for client_id, particle in enumerate(particles):
                    updated_particle, edge_model_params, transfer_time, client_output_size, fitness = EdgeUpdatePSO(
                        client_id, particle, copy.deepcopy(global_edge_model), client_dataloaders[client_id], t,
                        global_best_position, local_bw
                    )

                    particles[client_id] = updated_particle
                    edge_models.append(edge_model_params)
                    fitnesses.append(fitness)
                    round_transfer_time += transfer_time
                    round_client_output_size += client_output_size

                    if fitness > global_best_fitness:
                        global_best_fitness = fitness
                        global_best_position = {name: param.clone().detach() for name, param in
                                                updated_particle.best_position.items()}

                # Aggregate edge models
                global_edge_dict = calculate_weighted_average(edge_models, fitnesses)
                global_edge_model.load_state_dict(global_edge_dict)

                # Simulate edge to cloud transfer time
                edge_to_cloud_size = sum(p.nelement() * p.element_size() for p in global_edge_model.parameters())
                edge_to_cloud_time = simulate_transfer_time(edge_to_cloud_size, global_bw)
                round_transfer_time += edge_to_cloud_time

                # Calculate round communication cost
                round_comm_cost = calculate_communication_cost(
                    local_transfer_time=round_transfer_time,
                    edge_to_cloud_transfer_time=edge_to_cloud_time
                )

                total_round_comm_costs += round_comm_cost
                total_client_output_size += round_client_output_size
                total_edge_to_cloud_size += edge_to_cloud_size

                logger.info(f"Round {t} - Communication cost: {round_comm_cost:.2f} seconds")
                logger.info(f"Round {t} - Client output size: {round_client_output_size} bits")
                logger.info(f"Round {t} - Edge-to-cloud size: {edge_to_cloud_size} bits")

                # Evaluate global model
                global_client_model = ClientModel().to(device)
                global_client_model.load_state_dict(global_best_position)

                train_loss, train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(
                    global_client_model, global_edge_model, trainloader)

                global_losses.append(train_loss)
                global_accuracies.append(train_accuracy)
                global_precisions.append(train_precision)
                global_recalls.append(train_recall)
                global_f1_scores.append(train_f1)

                save_data(global_accuracies, '../Results/EdgeFedPSO_Accuracyb16e10.pkl')
                save_data(global_losses, '../Results/EdgeFedPSO_Lossesb16e10.pkl')
                save_data(global_precisions, '../Results/EdgeFedPSO_Precisionsb16e10.pkl')
                save_data(global_recalls, '../Results/EdgeFedPSO_Recallsb16e10.pkl')
                save_data(global_f1_scores, '../Results/EdgeFedPSO_f1Scoresb16e10.pkl')

                logger.info(f"Round {t}: Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, "
                            f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-Score: {train_f1:.4f}")

                test_loss, test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(
                    global_client_model, global_edge_model, testloader)

                logger.info(f"Round {t}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, "
                            f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-Score: {test_f1:.4f}")

            # Store the results for this bandwidth combination
            results.append({
                'Local Bandwidth (MB/s)': local_bw / (1024 * 1024),
                'Global Bandwidth (MB/s)': global_bw / (1024 * 1024),
                'Average Round Communication Cost (s)': total_round_comm_costs / NUM_ROUNDS,
                'Total Client Output Size (bits)': total_client_output_size,
                'Average Edge-to-Cloud Size (bits)': total_edge_to_cloud_size / NUM_ROUNDS
            })

    # Save all the results to a file
    save_data(results, '../Results/EdgeFedPSO_communication_costsb16e10.pkl')
    logger.info("Bandwidth simulation completed and results saved.")


# Run the GlobalAggregation procedure
if __name__ == "__main__":
    GlobalAggregationPSO()
