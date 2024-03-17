from sklearn.cluster import KMeans
import numpy as np
# Вхідна матриця
data = np.array([
    [0.6, 52, 2.1, 5, 19],
    [0.6, 33, 2.5, 5, 17],
    [1, 42, 2.2, 15, 15],
    [0.8, 45, 2.7, 10, 11],
    [0.2, 32, 1.9, 5, 18],
    [0.6, 25, 1.8, 20, 21],
    [0.9, 40, 1.5, 10, 20]
])

# Нормування
normalized_data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
print("Нормована матриця:")
print(normalized_data)

# Сума нормованих оцінок за кожним стовпчиком
sums = normalized_data.sum(axis=0)
print("\nСума нормованих оцінок за кожним стовпчиком:")
print(sums)

# Кількість кластерів
num_clusters = 2

# Використовуємо алгоритм K-means для кластеризації
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(normalized_data)

# Отримуємо мітки кластерів для кожного об'єкта
labels = kmeans.labels_
print("\nМітки кластерів:")
print(labels)

# Матриця відстаней
distances = kmeans.transform(normalized_data)
print("\nМатриця відстаней:")
print(distances)