import matplotlib.pyplot as plt
import pickle
import os

def load_data(filename):
    """Loads data from a pickle file."""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

num_rounds = 2
process_1 = load_data('Results/EdgeFedPSO_Accuracy.pkl')
# process_2 = load_data('/FedAvg/output_accuracy.pkl')
# plt.figure(figsize=(12, 5))
# # plt.subplot(1, 2, 1)
# # plt.plot(range(1, num_rounds + 1), global_losses, marker='o')
# # plt.title('Global Loss Over Communication Rounds')
# # plt.xlabel('Communication Rounds')
# # plt.ylabel('Loss')
#
# plt.subplot(1, 1, 1)
# plt.plot(range(1, num_rounds + 1), process_1, label='EdgeFed', marker='o')
# plt.plot(range(1, num_rounds + 1), process_2, label='EdgeFedPSO', marker='o')
# plt.title('Global Accuracy Over Communication Rounds')
# plt.xlabel('Communication Rounds')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
