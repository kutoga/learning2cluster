from __future__ import print_function
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np

cluster_count_probabilities = [
    0.1,
    0.2,
    0.6,
    0.1
]
cluster_4_index_probabilities = [
    # 0.05,
    0.3,
    0.7,
]
cluster_4_index_probabilities_2 = [
    0.25,
    0.6,
    0.15,
]

plt.subplots(2, 2, figsize=(7, 6))

cluster_counts = list(range(1, len(cluster_count_probabilities) + 1))
y_pos = np.arange(len(cluster_counts))
performance = cluster_count_probabilities
plt.subplot(2, 1, 1)
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, cluster_counts)
plt.ylim([0, 1])
plt.xlabel('k')
plt.ylabel('$P(k)$')
# plt.title('$P(k)$')

plt.subplot(2, 2, 3)
cluster_indices = list(range(1, 1 + len(cluster_4_index_probabilities)))
y_pos = cluster_indices
performance = cluster_4_index_probabilities
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, cluster_indices)
plt.ylim([0, 1])
plt.xlabel('$\ell$')
plt.ylabel('$P(\ell|x_2,k=2)$')
# plt.title('$P(l_i|x_i,k)$')
# plt.title('cluster index probabilities for $x_i$, given there are 4 clusters')

plt.subplot(2, 2, 4)
cluster_indices = list(range(1, 1 + len(cluster_4_index_probabilities_2)))
y_pos = cluster_indices
performance = cluster_4_index_probabilities_2
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, cluster_indices)
plt.ylim([0, 1])
plt.xlabel('$\ell$')
plt.ylabel('$P(\ell|x_2,k=3)$')
# plt.title('$P(l_j|x_j,k)$')
# plt.title('cluster index probabilities for $x_i$, given there are 4 clusters')

plt.tight_layout()
plt.show(block=True)

