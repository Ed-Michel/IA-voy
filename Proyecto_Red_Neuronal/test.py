import array
import math
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense  # Agregar esta línea para importar Dense
import csv
import matplotlib.pyplot as plt

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

#cargar archivos
trainning_file = "Proyecto_Red_Neuronal/Ts.csv"
evaluation_file = "Proyecto_Red_Neuronal/Vs.csv"

trainning_set = []
evaluation_set = []

with open(trainning_file, newline='') as archivo:
    lector_csv = csv.reader(archivo)
    
    # Ignorar la primera fila (encabezados)
    next(lector_csv)
    
    # Iterar sobre las filas restantes
    for fila in lector_csv:
        # Convertir los elementos de la fila a números y agregarlos a la lista
        fila_numeros = [float(valor) for valor in fila]
        trainning_set.append(fila_numeros)

with open(evaluation_file, newline='') as archivo:
    lector_csv = csv.reader(archivo)
    
    # Ignorar la primera fila (encabezados)
    next(lector_csv)
    
    # Iterar sobre las filas restantes
    for fila in lector_csv:
        # Convertir los elementos de la fila a números y agregarlos a la lista
        fila_numeros = [float(valor) for valor in fila]
        evaluation_set.append(fila_numeros)
    
trainning_set = numpy.array(trainning_set)
evaluation_set = numpy.array(evaluation_set)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def generate_individual():
    return [random.randint(0, 1) for _ in range(5)]

toolbox.register("indice", generate_individual)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indice)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#crear las redes neuronales

def evaluate(individual):
    posiciones = [indice for indice, valor in enumerate(individual) if valor == 1]
    neuronas = math.ceil(individual.count(1) * 1.5)

    modelo = tf.keras.Sequential()
    # capa de entrada
    modelo.add(Dense(units=5, activation='relu', input_shape=[len(posiciones)]))
    # capa oculta
    modelo.add(Dense(units = neuronas, activation= 'relu'))
    # capa de salida
    modelo.add(Dense(units=1, activation='relu'))

    modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)
    
    datos_entrada = trainning_set[:, posiciones]
    datos_salida = trainning_set[:, -1]

    datos_evaluacion = evaluation_set[:, posiciones]
    evaluacion_salida = evaluation_set[:, -1]
    
    historial = modelo.fit(datos_entrada, datos_salida, epochs=5, verbose=False)
    accuracy = modelo.evaluate(datos_evaluacion, evaluacion_salida, verbose=False)
    print("Precisión del modelo:", accuracy)
    return accuracy,

def trainning(individual):
    posiciones = [indice for indice, valor in enumerate(individual) if valor == 1]
    neuronas = math.ceil(individual.count(1) * 10)

    modelo = tf.keras.Sequential()
    # capa de entrada
    modelo.add(Dense(units=5, activation='relu', input_shape=[len(posiciones)]))
    # capa oculta
    modelo.add(Dense(units = neuronas, activation= 'relu'))
    # capa de salida
    modelo.add(Dense(units=1, activation='sigmoid'))

    modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
) 
    
    datos_entrada = trainning_set[:, posiciones]
    datos_salida = trainning_set[:, -1]

    datos_evaluacion = evaluation_set[:, posiciones]
    evaluacion_salida = evaluation_set[:, -1]
    
    historial = modelo.fit(datos_entrada, datos_salida, epochs=500, verbose=False)
    plt.xlabel("# Epoca")
    plt.ylabel("Magnitud de pérdida")
    plt.plot(historial.history["loss"])
    plt.show()
    accuracy = modelo.evaluate(datos_evaluacion, evaluacion_salida, verbose=False)
    print("Precisión del modelo:", accuracy)
    return accuracy,


toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", evaluate)


def main():
    random.seed(169)

    pop = toolbox.population(n=10)

    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 5, stats=stats, 
                        halloffame=hof)
    
    return pop, stats, hof

if __name__ == "__main__":
    pop,stats,hof=main()
    evaluado =[]
    for individual in hof:
        evaluacion = trainning(individual)
        evaluado.append((individual, evaluacion))
    
    evaluado.sort(key=lambda x: x[1])
    print(evaluado)