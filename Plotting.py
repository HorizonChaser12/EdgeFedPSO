import matplotlib.pyplot as plt

import data_storage
from data_storage import load_data

num_rounds = 5
process_1 = data_storage.load_data('output_a.pkl')
process_2 = data_storage.load_data('output_b.pkl')
plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(range(1, num_rounds + 1), global_losses, marker='o')
# plt.title('Global Loss Over Communication Rounds')
# plt.xlabel('Communication Rounds')
# plt.ylabel('Loss')

plt.subplot(1, 1, 1)
plt.plot(range(1, num_rounds + 1), process_1, label='EdgeFed', marker='o')
plt.plot(range(1, num_rounds + 1), process_2, label='EdgeFedPSO', marker='o')
plt.title('Global Accuracy Over Communication Rounds')
plt.xlabel('Communication Rounds')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
