# EdgeFedPSO (Edge Federated Particle Swarm Optimization)

EdgeFedPSO is an innovative project that implements Federated Learning using Particle Swarm Optimization (PSO) on edge devices. It aims to train machine learning models collaboratively across edge devices while preserving data privacy and minimizing communication overhead.It provides the code for the Research Paper **"An Optimized Machine Learning Technique on Edge Environment : EdgeFedPSO"** 

> :warning: **This research has been halted.**: If you want to use this research,you can give us a reference.Hoping you guys get this posted in some Research Papers.

## Description

EdgeFedPSO leverages the power of federated learning to train machine learning models across edge devices, such as smartphones and IoT devices, without the need to centralize data. The project employs the Particle Swarm Optimization algorithm to optimize model parameters locally on edge devices while periodically exchanging information with a central server to aggregate global updates.

## Updates

<details>
<summary>
Updates 2
</summary>

### **Updates in EdgeFedPSO:** ###
1. **Model Architecture Enhancements**:
   - Introduced a split CNN architecture for improved training efficiency. The `ClientModel` and `EdgeModel` are now defined to separately handle local and edge computations.
   - Added dropout layers to reduce overfitting in both models.
2. **Particle Swarm Optimization (PSO) Implementation**:
   - Enhanced the `Particle` class with methods for updating velocity, position, and best fitness using PSO.
   - Implemented a `GlobalAggregationPSO` method that aggregates model parameters from multiple clients based on fitness scores, improving overall model performance.
3. **Non-IID Data Simulation**:
   - Added a function `create_non_iid_data` to generate non-IID datasets for clients, enhancing realism in federated learning scenarios.
4. **Bandwidth Simulation**:
   - Integrated bandwidth simulation for local and global transfers with configurable scenarios (best, worst, and default). The `simulate_transfer_time` function now models the time taken for data transfer based on client and edge server bandwidth.
5. **Particle Diversity**: 
   - The PSO algorithm now initializes particles with small random noise for increased diversity.
6. **Communication Cost Calculation**:
   - Implemented a function to calculate communication costs for each round, helping to analyze the efficiency of the federated learning process.

### **Updates in EdgeFed:**
1. **Non-IID Data Distribution**: 
   - Implemented a Dirichlet distribution-based method for creating non-IID data across clients, providing a more realistic scenario.
2. **Network Simulation**: 
   - Added bandwidth simulation to model data transfer between clients, edge servers, and the cloud.
3. **Learning Rate Decay**: 
   - Introduced a learning rate decay mechanism to improve convergence.
4. **Model Architecture**: 
   - Updated the CNN model architecture, splitting it into client and edge components for more efficient federated learning.
</details>


<details>
<summary>
Updates 1
</summary>


### **Updates in EdgeFedPSO:**
1. **PSO-based Model Update**: 
   - The Particle Swarm Optimization (PSO) algorithm has been integrated into the client-side model updates. Instead of traditional gradient-based updates, each client employs      multiple particles to explore better model parameters, combining cognitive and social terms for better convergence.
   - New helper functions to initialize particles and calculate fitness.
   - Velocity updates now include position clipping for better control, and diverse particles are initialized with noise.
2. **Improved Metrics and Aggregation**:
   - Clients now send updated models with precision, recall, and F1-score for enhanced evaluation.
   - Weighted averaging of model parameters uses PSO to fine-tune global model updates.
3. **Global Stopping Condition**:
   - Introduced accuracy threshold (95%) for early stopping during global aggregation.
4. **Efficiency Improvements**:
   - Refined learning rate adjustment and optimized PSO update process for faster convergence.

### **Updates in EdgeFed:**
1. **Dataset Splitting:**
   - In the updated version, `client_dataloaders` are created using subsets of the MNIST dataset for each edge server. This ensures that each server gets a unique portion of the dataset, reflecting a more realistic federated learning setup.
2. **Weighted Average of Model Parameters:**
   - The aggregation function is more explicit in calculating weighted averages of model parameters across clients based on their data size, improving the effectiveness of global model updates.
3. **Global Model Update:**
   - The new implementation ensures that the global model is updated directly using the aggregated local models’ state_dicts, improving synchronization across clients.

### **Misc Updates:**
1. **Enhanced Logging and Metrics:**
   - The updated code logs training and test set metrics (loss, accuracy, precision, recall, F1-score) for each communication round. This provides more visibility into model performance during training.
2. **Storage of Results:**
   - Enhanced result-saving capabilities, storing evaluation metrics (accuracy, loss, precision, recall, F1-score) after each round in `.pkl` files, making it easier to analyze training progression later.
3. **Evaluation on Test Set:**
   - The updated version evaluates both training and test data during each round, providing a more comprehensive performance analysis.
4. **Device Configuration:**
   - Added `torch.device` for utilizing GPU (`cuda`) if available, improving the performance for training on large datasets.

</details>

## Installation

To use the EdgeFedPSO project, follow these steps:

1. Clone the repository: `git clone https://github.com/your_username/EdgeFedPSO.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Set up the edge devices where you want to deploy the federated learning framework.
4. Configure the parameters such as the number of edge servers, clients per server, local batch size, learning rate, etc., according to your requirements.
5. Prepare the dataset for federated learning. Ensure that it is divided into subsets for each edge device.
6. Run the project to start federated learning: `python edge_fed_pso.py`.
7. Monitor the training progress and global model updates using the logger that prints the particular results.


## Troubleshooting

- If the training process does not converge or encounters errors, try adjusting hyperparameters such as learning rate, batch size, or PSO parameters.
- Ensure that the edge devices have sufficient computational resources and data samples for training.
- Debug any issues related to communication between edge devices and the central server.

## Dependencies
The main dependencies for this project are:
   - PyTorch (1.8.0+)
   - NumPy (1.19.0+)
   - scikit-learn (0.24.0+)
   - torchvision (0.9.0+)

For a complete list of dependencies, please refer to the `requirements.txt` file.

## Future Development

This project is actively under development, and future plans include:

- Implement more sophisticated non-IID data distribution methods.
- Explore adaptive PSO parameters based on system performance.
- Incorporate privacy-preserving techniques such as differential privacy.
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
