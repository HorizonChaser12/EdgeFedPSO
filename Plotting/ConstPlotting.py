import matplotlib.pyplot as plt
import data_storage

# Load the data from pickle files
process_1 = data_storage.load_data('../Results/EdgeFedPSO_Accuracy.pkl')

process_2 = data_storage.load_data('../Results/EdgeFed_Accuracy.pkl')

num_rounds = 10  # Assuming the number of rounds is 5

# Plotting
plt.figure(figsize=(18, 5))

# # F1-Score
# plt.subplot(1, 3, 1)
# plt.plot(range(1, num_rounds + 1), process_1, label='EdgeFed', marker='o')
# # plt.plot(range(1, num_rounds + 1), process_2, label='Research', marker='o')
# plt.title('Global F1-Score Over Communication Rounds')
# plt.xlabel('Communication Rounds')
# plt.ylabel('F1-Score')
# plt.legend()

# Recall
plt.subplot(1, 1, 1)
plt.plot(range(1, num_rounds + 1), process_1, label='Research', marker='o')
plt.plot(range(1, num_rounds + 1), process_2, label='EdgeFed', marker='o')
plt.title('Global Recall Over Communication Rounds')
plt.xlabel('Communication Rounds')
plt.ylabel('Accuracy')
plt.legend()

# Precision
# plt.subplot(1, 2, 2)
# # plt.plot(range(1, num_rounds + 1), process_1, label='EdgeFed', marker='o')
# plt.plot(range(1, num_rounds + 1), process_2, label='Research', marker='o')
# plt.title('Global Precision Over Communication Rounds')
# plt.xlabel('Communication Rounds')
# plt.ylabel('Precision')
# plt.legend()

plt.tight_layout()
plt.show()