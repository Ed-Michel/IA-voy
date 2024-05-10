import numpy as np
import tensorflow as tf
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generar datos sintéticos
n_samples = 300
n_features = 2
n_clusters = 3
X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=42)

# Normalizar los datos
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

# Definir el modelo K-Means como una red neuronal con una sola capa
class KMeansModel(tf.keras.Model):
    def __init__(self, n_clusters):
        super(KMeansModel, self).__init__()
        self.n_clusters = n_clusters
        self.cluster_centers = tf.Variable(initial_value=tf.random.normal(shape=(n_clusters, n_features)))

    def call(self, inputs):
        expanded_vectors = tf.expand_dims(inputs, axis=1)
        expanded_centers = tf.expand_dims(self.cluster_centers, axis=0)
        distances = tf.reduce_sum(tf.square(expanded_vectors - expanded_centers), axis=-1)
        return tf.argmin(distances, axis=-1)

# Función de pérdida para K-Means
def kmeans_loss(y_true, y_pred, cluster_centers):
    expanded_centers = tf.expand_dims(cluster_centers, axis=0)
    cluster_centers_loss = tf.reduce_sum(tf.square(expanded_centers - expanded_centers), axis=(1, 2))
    return tf.reduce_mean(cluster_centers_loss)

# Crear el modelo
model = KMeansModel(n_clusters)

# Optimizador
optimizer = tf.keras.optimizers.Adam()

# Ciclo de entrenamiento
for epoch in range(500):
    with tf.GradientTape() as tape:
        y_pred = model(X_normalized)
        loss = kmeans_loss(None, y_pred, model.cluster_centers)
    gradients = tape.gradient(loss, model.trainable_variables)
    if not all(grad is None for grad in gradients):
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Predecir las etiquetas de los clústeres
y_pred = model.predict(X_normalized)

# Visualizar los resultados
plt.figure(figsize=(8, 6))
plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=y_pred, cmap='viridis')
plt.scatter(model.cluster_centers[:, 0], model.cluster_centers[:, 1], marker='x', s=100, c='red')
plt.title('K-Means clustering with Neural Network')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.grid(True)
plt.show()