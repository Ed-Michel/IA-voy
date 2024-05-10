import array
import math
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense
import csv
import matplotlib.pyplot as plt
from tensorflow.keras.initializers import TruncatedNormal
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# Cargar archivo CSV
file_path = "Proyecto_Red_Neuronal\entrada_y_salida_sin_vacios.csv"

dataset = []

with open(file_path, newline='') as archivo:
    lector_csv = csv.reader(archivo)
    next(lector_csv)  # Ignorar la primera fila (encabezados)
    for fila in lector_csv:
        fila_numeros = [float(valor) for valor in fila]  # Convertir los elementos de la fila a números
        dataset.append(fila_numeros)

dataset = np.array(dataset)  # Convertir la lista a un array numpy

# Normalizar los datos
scaler = MinMaxScaler()
dataset_normalized = scaler.fit_transform(dataset)

# Dividir los datos en conjunto de entrenamiento y conjunto de evaluación (70% training, 30% evaluation)
trainning_set, evaluation_set = train_test_split(dataset_normalized, test_size=0.3, random_state=42)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def generate_individual():
    return [random.randint(0, 1) for _ in range(5)]

toolbox.register("indice", generate_individual)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indice)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    posiciones = [indice for indice, valor in enumerate(individual) if valor == 1]
    neuronas = math.ceil(individual.count(1) * 3)

    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=[len(posiciones)]),
        tf.keras.layers.Dense(units=neuronas, activation='tanh', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1)),
        tf.keras.layers.Dense(units=1, activation='linear', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1))
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),  # Reducimos la tasa de aprendizaje
        loss='mean_squared_error',
        metrics=['mse']
    )

    datos_entrada = trainning_set[:, posiciones]
    datos_salida = trainning_set[:, -1]

    datos_evaluacion = evaluation_set[:, posiciones]
    evaluacion_salida_evaluate = evaluation_set[:, -1]

    historial = modelo.fit(datos_entrada, datos_salida, epochs=5, batch_size=3500, verbose=False)
    mse = modelo.evaluate(datos_evaluacion, evaluacion_salida_evaluate, verbose=False)[1]  # Extraemos el MSE
    
    print("MSE del modelo:", mse)
        
    return mse,

def trainning(individual):
    posiciones = [indice for indice, valor in enumerate(individual) if valor == 1]
    neuronas = math.ceil(individual.count(1) * 10)

    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=[len(posiciones)]),
        tf.keras.layers.Dense(units=neuronas, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),  # Reducimos la tasa de aprendizaje
        loss='mean_squared_error',
        metrics=['mse']
    )

    datos_entrada = trainning_set[:, posiciones]
    datos_salida = trainning_set[:, -1]

    datos_evaluacion = evaluation_set[:, posiciones]
    evaluacion_salida_evaluate = evaluation_set[:, -1]

    history = modelo.fit(datos_entrada, datos_salida, epochs=500, batch_size=3500, verbose=False)
    mse = modelo.evaluate(datos_evaluacion, evaluacion_salida_evaluate, verbose=False)[1] 
    print("MSE del modelo:", mse)
    
    plt.plot(history.history['mse'], label='Training MSE')
    plt.title('Model MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    
    return mse,

toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.03)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", evaluate)

def main():
    random.seed(169)

    pop = toolbox.population(n=10)

    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 5, stats=stats, halloffame=hof)

    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof = main()
    evaluado = []
    for individual in hof:
        evaluacion = trainning(individual)
        evaluado.append((individual, evaluacion))

    evaluado.sort(key=lambda x: x[1])
    print(evaluado)
    
    # Obtener el mejor individuo
    mejor_individuo = hof[0]

    # Obtener las posiciones del mejor individuo
    posiciones_mejor_individuo = [indice for indice, valor in enumerate(mejor_individuo) if valor == 1]

    # Crear el modelo con las características del mejor individuo
    modelo_mejor_individuo = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=[len(posiciones_mejor_individuo)]),
        tf.keras.layers.Dense(units=len(posiciones_mejor_individuo) * 3, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])

    modelo_mejor_individuo.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),  # Reducimos la tasa de aprendizaje
        loss='mean_squared_error',
        metrics=['mse']
    )

    # Entrenar el modelo con el conjunto de entrenamiento completo
    datos_entrada = trainning_set[:, posiciones_mejor_individuo]
    datos_salida = trainning_set[:, -1]
    history = modelo_mejor_individuo.fit(datos_entrada, datos_salida, epochs=500, batch_size=3500, verbose=False)

    # Obtener las predicciones del modelo en el conjunto de evaluación
    datos_evaluacion = evaluation_set[:, posiciones_mejor_individuo]
    predicciones_evaluacion = modelo_mejor_individuo.predict(datos_evaluacion)

    # Graficar las predicciones junto con los valores reales de salida
    plt.plot(evaluation_set[:, -1], label='Actual Output')
    plt.plot(predicciones_evaluacion, label='Predicted Output')
    plt.title('Actual vs Predicted Output')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.show()