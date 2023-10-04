import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

np.random.seed(0)
X = np.random.rand(100, 2)

n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, n_init=10)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

k = 5
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(X)

colors = ['r', 'g', 'b']

plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], c=colors[i], label=f'Кластер {i}')

random_point = np.random.rand(1, 2)
plt.scatter(random_point[:, 0], random_point[:, 1], c='m', marker='x', s=100, label='Случайная точка')

distances, indices = neigh.kneighbors(random_point)

print("Ближайшие соседи для случайной точки:")
for i in range(k):
    neighbor_index = indices[0][i]
    cluster_label = y_kmeans[neighbor_index]
    print(f"Сосед {i + 1}: Point {neighbor_index} (Кластер {cluster_label}) with distance {distances[0][i]}")

random_point_cluster = y_kmeans[kmeans.predict(random_point)[0]]

print(f"Наиболее часто встречающийся кластер для случайной точки: {random_point_cluster + 1}")

plt.title('K-Means Clustering')
plt.legend()
plt.show()
