# EdgeFedPSO (Edge Federated Particle Swarm Optimization)

EdgeFedPSO is an innovative project that implements Federated Learning using Particle Swarm Optimization (PSO) on edge devices. It aims to train machine learning models collaboratively across edge devices while preserving data privacy and minimizing communication overhead.It provides the code for the Research Paper **"An Optimized Machine Learning Technique on Edge Environment : EdgeFedPSO"** 

## Description

EdgeFedPSO leverages the power of federated learning to train machine learning models across edge devices, such as smartphones and IoT devices, without the need to centralize data. The project employs the Particle Swarm Optimization algorithm to optimize model parameters locally on edge devices while periodically exchanging information with a central server to aggregate global updates.

## Updates

Updates in EdgeFedPSO:
1. PSO-based Model Update: The Particle Swarm Optimization (PSO) algorithm has been integrated into the client-side model updates. Instead of traditional gradient-based updates, each client employs multiple particles to explore better model parameters, combining cognitive and social terms for better convergence.
2. Improved Metrics and Aggregation: Clients now send updated models with precision, recall, and F1-score for enhanced evaluation.
Weighted averaging of model parameters uses PSO to fine-tune global model updates.
3. Global Stopping Condition:Introduced accuracy threshold (95%) for early stopping during global aggregation.
4. Efficiency Improvements:Refined learning rate adjustment and optimized PSO update process for faster convergence.










## Key Features

- Federated learning for decentralized model training
- Particle Swarm Optimization for local model optimization
- Support for various machine learning tasks (classification, regression, etc.)
- Privacy-preserving approach by keeping data on edge devices
- Minimization of communication overhead by aggregating global updates periodically

## Installation

To use the EdgeFedPSO project, follow these steps:

1. Clone the repository: `git clone https://github.com/your_username/EdgeFedPSO.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Set up the edge devices where you want to deploy the federated learning framework.
4. Configure the parameters such as the number of edge servers, clients per server, local batch size, learning rate, etc., according to your requirements.
5. Prepare the dataset for federated learning. Ensure that it is divided into subsets for each edge device.
6. Run the project to start federated learning: `python edge_fed_pso.py`.
7. Monitor the training progress and global model updates.

## Usage

1. Define your machine learning model architecture and objective function in the `Model` class.
2. Implement the `calculate_gradient` function to compute local gradients on each edge device.
3. Execute the `EdgeUpdate` function to perform local model updates on each edge device.
4. Run the `GlobalAggregation` function to aggregate global model updates periodically.

## Troubleshooting

- If the training process does not converge or encounters errors, try adjusting hyperparameters such as learning rate, batch size, or PSO parameters.
- Ensure that the edge devices have sufficient computational resources and data samples for training.
- Debug any issues related to communication between edge devices and the central server.

## Future Development

This project is actively under development, and future plans include:

- Implementing advanced federated learning algorithms and optimization techniques
- Extending support for different types of machine learning tasks and models
- Enhancing security measures to protect data privacy during federated learning
- Integrating with edge computing frameworks for improved performance and scalability

Contributions and suggestions are welcome! Feel free to open issues and submit pull requests to help improve the project.

This is a made with the joined efforts of my colleagues :
1. Satya Swarup Sahu
2. Akshit Shaha
3. Amarnath Ghosh

## Acknowledgements

We would like to express our gratitude to the developers and contributors of the following libraries and frameworks that made this project possible:

- PyTorch
- NumPy
- and other open-source projects used in this project.
