import numpy as np
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
        model.zero_grad()  # Add this line to zero out the gradients
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


# New helper function to initialize diverse particles
def initialize_diverse_particles(global_model, num_particles):
    particles = []
    for _ in range(num_particles):
        particle = {'model': copy.deepcopy(global_model.state_dict()), 'score': float('inf'),
                    'velocity': {key: torch.zeros_like(value) for key, value in global_model.state_dict().items()},
                    'best_model': copy.deepcopy(global_model.state_dict()), 'best_score': float('inf')}
        # Add small random noise to parameters
        for key in particle['model']:
            particle['model'][key] += torch.randn_like(particle['model'][key]) * 0.01
        particles.append(particle)
    return particles


# New helper function to calculate fitness
def calculate_fitness(model, dataloader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / len(dataloader)
    accuracy = (total_correct / total_samples) * 100.0

    # Combine loss and accuracy for fitness (lower is better)
    fitness = average_loss - 0.1 * accuracy
    return fitness, average_loss, accuracy


# Modified PSOUpdate function
def PSOUpdate(client_id, global_model, dataloader, global_best_score, global_best_model):
    num_particles = 5
    inertia_weight = 0.7
    cognitive_coeff = 1.5
    social_coeff = 2.0
    velocity_scale = 0.01
    velocity_limit = 0.1

    particles = initialize_diverse_particles(global_model, num_particles)

    for epoch in range(1, E + 1):
        for particle in particles:
            local_model = copy.deepcopy(global_model)
            local_model.load_state_dict(particle['model'])
            local_model.train()

            optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)

            # Local training
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = local_model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                optimizer.step()

            # Update particle's model after training
            particle['model'] = copy.deepcopy(local_model.state_dict())

            # Calculate fitness
            fitness, average_loss, accuracy = calculate_fitness(local_model, dataloader, device)

            # Update personal best
            if fitness < particle['best_score']:
                particle['best_model'] = copy.deepcopy(particle['model'])
                particle['best_score'] = fitness

            # Update global best
            if fitness < global_best_score:
                global_best_model.load_state_dict(copy.deepcopy(particle['model']))
                global_best_score = fitness

        # PSO velocity and position update
        for particle in particles:
            r1, r2 = np.random.rand(), np.random.rand()
            for key in particle['velocity']:
                particle['velocity'][key] = (inertia_weight * particle['velocity'][key] +
                                             cognitive_coeff * r1 * (
                                                         particle['best_model'][key] - particle['model'][key]) +
                                             social_coeff * r2 * (
                                                         global_best_model.state_dict()[key] - particle['model'][key]))

                # Clip velocity
                particle['velocity'][key] = torch.clamp(particle['velocity'][key], -velocity_limit, velocity_limit)

                # Update model parameters with scaled velocity
                particle['model'][key] += velocity_scale * particle['velocity'][key]

        logger.info(
            f"Client {client_id} - Epoch {epoch}\tAccuracy: {accuracy:.2f}%\tLoss: {average_loss:.4f}")

    best_particle = min(particles, key=lambda x: x['best_score'])
    return best_particle['model'], global_best_score, global_best_model


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
testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
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


# Modified GlobalAggregation function
def GlobalAggregation():
    global_best_model = CNNModel().to(device)
    global_best_score = float('inf')

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

    optimizer = optim.SGD(global_best_model.parameters(), lr=learning_rate)

    for t in range(1, num_rounds + 1):
        weighted_params = []

        # PSO update for each client
        for client_id in range(num_edge_servers):
            local_model_params, global_best_score, global_best_model = PSOUpdate(
                client_id, global_best_model, client_dataloaders[client_id], global_best_score, global_best_model)
            weighted_params.append(local_model_params)

        # Weighted average aggregation of model parameters
        weights = [1.0] * num_edge_servers
        weighted_average_params = calculate_weighted_average(weighted_params, weights)
        global_best_model.load_state_dict(weighted_average_params)

        # Evaluate on training data
        global_loss, global_accuracy, true_labels, predictions = calculate_gradient(global_best_model, trainloader)
        precision, recall, f1 = calculate_metrics(true_labels, predictions)

        # Log training metrics
        global_losses.append(global_loss)
        global_accuracies.append(global_accuracy)
        global_precisions.append(precision)
        global_recalls.append(recall)
        global_f1_scores.append(f1)

        save_data(global_accuracies, '../Results/EdgeFedPSO_Accuracy.pkl')
        save_data(global_losses, '../Results/EdgeFedPSO_Losses.pkl')
        save_data(global_precisions, '../Results/EdgeFedPSO_Precisions.pkl')
        save_data(global_recalls, '../Results/EdgeFedPSO_Recalls.pkl')
        save_data(global_f1_scores, '../Results/EdgeFedPSO_f1Scores.pkl')

        logger.info(f"Round {t}: Training Loss: {global_loss:.4f}, Training Accuracy: {global_accuracy:.2f}%, "
                    f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

        # Evaluate on test data
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_global_model(global_best_model, testloader)

        logger.info(f"Round {t}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, "
                    f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-Score: {test_f1:.4f}")

        # Adjust learning rate
        adjust_learning_rate(optimizer)

GlobalAggregation()