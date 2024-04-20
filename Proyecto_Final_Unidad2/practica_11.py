import math
import numpy as np
import random

class DNA:
    def __init__(self, target, mutation_rate, n_individuals, n_selection, n_generations, verbose=True):
        self.target = target
        self.mutation_rate = mutation_rate
        self.n_individuals = n_individuals
        self.n_selection = n_selection
        self.n_generations = n_generations
        self.verbose = verbose

    # creación de un individuo
    def create_individual(self, max=15):
        possible_values = [i for i in range(1, max) if i not in [0, 4, 7, 12]]
        individual = random.sample(possible_values, 7)
        return individual

    # creación de toda la población, iterando en la creación de los individuos
    def create_population(self):
        return [self.create_individual() for _ in range(self.n_individuals)]
    
    # cálculo de la distancia Euclidiana
    def distancia(self, punto1, punto2):
        """ 
        Calcula la distancia euclidiana entre dos puntos en un espacio n-dimensional.

        Args:
        punto1 (list): Coordenadas del primer punto.
        punto2 (list): Coordenadas del segundo punto.

        Returns:
        float: La distancia euclidiana entre los dos puntos.
        """
        if len(punto1) != len(punto2):
            raise ValueError("Los puntos deben tener la misma cantidad de dimensiones.")
    
        suma_cuadrados = sum((q - p) ** 2 for p, q in zip(punto1, punto2))
        distancia = math.sqrt(suma_cuadrados)
        return distancia
    
    # función fitness en dónde se realiza el recorrido establecido en el proyecto para calcular la distancia recorrida
    def fitness(self, individuo):
        # Diccionario de coordenadas
        coordenadas = {
            0: [1, 7],
            1: [3, 7],
            2: [5, 7],
            3: [7, 7],
            4: [1, 5],
            5: [3, 5],
            6: [5, 5],
            7: [7, 5],
            8: [1, 3],
            9: [3, 3],
            10: [5, 3],
            11: [7, 3],
            12: [1, 1],
            13: [3, 1],
            14: [5, 1],
            15: [7, 1]
        }

        # Calcular la distancia total recorrida
        fitness = 0
        cont = 1
        while(cont < 9):
            if(cont == 1):
                     fitness +=  2 * (self.distancia(coordenadas[individuo[0]], coordenadas[0]))
            if(cont == 2):
                 fitness += self.distancia(coordenadas[individuo[0]], coordenadas[individuo[3]])
            if(cont == 3):
                 fitness += self.distancia(coordenadas[individuo[3]], coordenadas[individuo[4]])
            if(cont == 4):
                 fitness += self.distancia(coordenadas[individuo[4]], coordenadas[individuo[2]])
            if(cont == 5):
                 fitness += 2* (self.distancia(coordenadas[individuo[3]], coordenadas[individuo[2]]))
            if(cont == 6):
                 fitness += self.distancia(coordenadas[individuo[2]], coordenadas[individuo[0]])
            if(cont == 7):
                 fitness += 2* ( self.distancia(coordenadas[4], coordenadas[individuo[0]]))
            if(cont == 8):
                 fitness += 2* ( self.distancia(coordenadas[2], coordenadas[individuo[0]]))
            if(cont == 9):
                 fitness += self.distancia(coordenadas[individuo[0]], coordenadas[7])
            cont += 1
        return fitness

    # se selecciona al mejor individuo de la población
    def selection(self, population):
        scores = [(self.fitness(i), i) for i in population]
        scores = [i[1] for i in sorted(scores)]

        print(scores)
        return scores[len(scores)-self.n_selection:]
    
    # se hace el cruce para generar nuevos individuos
    def reproduction(self, population, selected):
        children = []
        for _ in range(self.n_individuals):
            father = random.sample(selected, 2)
            child = father[0][:]

            # Seleccionar valores únicos para el hijo
            for i in range(len(child)):
                if random.random() < 0.5:
                    while True:
                        new_value = random.choice([x for x in range(1, 16) if x not in child])
                        if new_value not in child:
                            child[i] = new_value
                            break
            children.append(child)
        return children
    
    # se realizan las variaciones para añadir una pizca de aleatoriedad
    def mutation(self, population):
        for i in range(len(population)):
            if random.random() <= self.mutation_rate:
                point = np.random.randint(len(self.target))
                new_value = np.random.randint(1, 16)
            
                # Asegurar que el nuevo valor no sea 0, 4, 7 o 12
                while new_value in [0, 4, 7, 12]:
                    new_value = np.random.randint(1, 16)
            
                # Asegurar que el nuevo valor sea diferente de los valores existentes en el individuo
                while new_value in population[i]:
                    new_value = np.random.randint(1, 16)

                population[i][point] = new_value
        return population

    def run_geneticalgo(self):
        population = self.create_population()

        for i in range(self.n_generations):

            if self.verbose:
                print('_')
                print('Generacion: ', i)
                print('Poblacion', population)
                print()

            selected = self.selection(population)
            population = self.reproduction(population, selected)
            population = self.mutation(population)

def main():
    target = [1,0,0,1,1,0,0]
    model = DNA(
        target = target,
        mutation_rate = 0.3,
        n_individuals = 100,
        n_selection = 50,
        n_generations = 50,
        verbose=True)
    model.run_geneticalgo()

if __name__ == "__main__":
    main()