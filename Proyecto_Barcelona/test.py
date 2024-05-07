import tensorflow as tf
# import csv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.initializers import TruncatedNormal
import time

# Cargar archivos
trainning_file = "Proyecto_Barcelona\Ts.csv"
evaluation_file = "Proyecto_Barcelona\Vs.csv"

trainning_set = np.loadtxt(trainning_file, delimiter=',', skiprows=1)
evaluation_set = np.loadtxt(evaluation_file, delimiter=',', skiprows=1)

# Separar datos de entrada y salida
trainning_input = trainning_set[:, 6:7]  
trainning_output = trainning_set[:, 6] 

evaluation_input = evaluation_set[:, 6:7] 
evaluation_output = evaluation_set[:, 6]

# Definir arquitectura de la red neuronal
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, activation='tanh', input_shape=(trainning_input.shape[1],), kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1)),
    tf.keras.layers.Dense(units=32, activation='tanh'),
    tf.keras.layers.Dense(units=1, activation='linear', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1))
])

# Compilar el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss='mean_squared_error',
    metrics=['mse']
)

# Entrenar el modelo
start_time = time.time()
history = modelo.fit(trainning_input, trainning_output, epochs=500, batch_size=3500, verbose=False)
elapsed_time = time.time() - start_time

# Realizar predicciones utilizando el conjunto de datos de evaluación
predictions = modelo.predict(evaluation_input)

# Graficar las predicciones vs los valores reales
plt.figure(figsize=(10, 6))
plt.plot(evaluation_output, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('Data Point')
plt.ylabel('Value')
plt.legend()
plt.show()

# Evaluar el modelo
mse = modelo.evaluate(evaluation_input, evaluation_output, verbose=False)[1]
print("MSE del modelo:", mse)
print("Tiempo de entrenamiento:", elapsed_time, "segundos")

# Graficar el error cuadrático medio durante el entrenamiento
plt.plot(history.history['mse'], label='Training MSE')
plt.title('Model MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()