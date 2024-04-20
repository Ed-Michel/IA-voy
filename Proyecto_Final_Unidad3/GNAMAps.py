import numpy as np
import random
import osmnx as ox


PuntoInicioyFin = [24.750986500213106, -107.45732566971502]  # cedis coppel
PuntosInteres = [
    [24.763531, -107.453989],  # coppel las torres
    [24.747421, -107.426716],  # coppel barrancos
    [24.756552, -107.403092],  # coppel san isidro
    [24.777955, -107.408095],  # coppel clouthier
    [24.739089, -107.393110],  # Coppel canada Clouthier
    [24.777126, -107.432649],  # Coppel bodega Remate
    [24.795033, -107.410673],  # Coppel Plaza fiesta
    [24.797711, -107.423402],  # Coppel la ceiba
    [24.769247, -107.371943],  # Coppel abastos
    [24.798349, -107.393610],  # Coppel Obregon
    [24.806291588486456, -107.39628727881541],  # Coppel Escobedo
    [24.80085076980745, -107.39122225493001],  # Coppel andrade
    [24.816005114020253, -107.42619802671574],  # Coppel Humaya
    [24.808862282043563, -107.39730792212923],  # Coppel Angel flores
    [24.808594949094665, -107.39575474973914],  # Coppel Portales
    [24.81342495074406, -107.40072622765123],  # Coppel plaza Forum
    [24.801446389514645, -107.36533391787977],  # Coppel Madero
    [24.834542368635553, -107.39806426389583],  # Coppel las flores
    [24.8259647322571, -107.42613555285189],  # Coppel sendero
]

graph = ox.graph_from_place("Culiacan Sinaloa", network_type='drive')
coordenadas_ruta = []


class DNA:
    def __init__(self, target, mutation_rate, n_individuals, n_selection, n_generations, verbose=True):
        self.target = target
        self.mutation_rate = mutation_rate
        self.n_individuals = n_individuals
        self.n_selection = n_selection
        self.n_generations = n_generations
        self.verbose = verbose



    def create_individual(self):
        individual = random.sample(self.target, len(self.target))
        print("Individuo generado:", individual)
        return individual
    
    def create_population(self):
        return [self.create_individual() for _ in range(self.n_individuals)]
    
    
    def fitness(self, individuo):
        fitness = 0
        # calcular desde el punto incio al primer punto de interes:
        nodo_inicio = ox.nearest_nodes(graph, PuntoInicioyFin[1], PuntoInicioyFin[0])
        nodo_fin = ox.nearest_nodes(graph, individuo[0][1], individuo[0][0])
        rutacorta = ox.shortest_path(graph, nodo_inicio, nodo_fin, weight='length')
        fitness += sum(ox.utils_graph.get_route_edge_attributes(graph, rutacorta, 'length'))

        # se calculan los puntos intermedios
        for i in range(len(PuntosInteres) - 1):  # iterar solo hasta el penúltimo punto
            nodo_inicio = ox.nearest_nodes(graph, individuo[i][1], individuo[i][0])
            nodo_fin = ox.nearest_nodes(graph, individuo[i + 1][1], individuo[i + 1][0])
            rutacorta = ox.shortest_path(graph, nodo_inicio, nodo_fin, weight='length')

            fitness += sum(ox.utils_graph.get_route_edge_attributes(graph, rutacorta, 'length'))
            print(fitness)

# calcular desde el punto final al ultimo punto interes:
        nodo_fin = ox.nearest_nodes(graph, PuntoInicioyFin[1], PuntoInicioyFin[0])
        nodo_inicio = ox.nearest_nodes(graph, individuo[len(individuo)-1][1], individuo[len(individuo)-1][0])
        rutacorta = ox.shortest_path(graph, nodo_inicio, nodo_fin, weight='length')

        fitness += sum(ox.utils_graph.get_route_edge_attributes(graph, rutacorta, 'length'))
        

        return fitness

    
    def selection(self, population):
        scores = [(self.fitness(i), i) for i in population]
        scores = [i[1] for i in sorted(scores)]

        print(scores[0])
        return scores[len(scores)-self.n_selection:]
    
    def reproduction(self, population, selected):
        children = []

    # Iterar sobre la cantidad de individuos que quieres generar
        for _ in range(self.n_individuals):
        # Seleccionar aleatoriamente dos individuos de los seleccionados
            parent1, parent2 = random.sample(selected, 2)
        # Crear un hijo combinando los dos individuos seleccionados
            child = self.reproduce_individuals(parent1, parent2)
            children.append(child)  # Agregar el hijo a la lista de hijos

        return children

    def reproduce_individuals(self, parent1, parent2):
        """
        Función que combina dos individuos para producir un nuevo hijo.

        Args:
        - parent1 (list): Primer individuo seleccionado para reproducción.
        - parent2 (list): Segundo individuo seleccionado para reproducción.

        Returns:
        - child (list): Nuevo individuo generado como hijo de los dos padres proporcionados.
        """
    # Combinar el orden de visita de ambos padres para generar al hijo
        child = []

        for point in parent1:
            if point in parent2 and point in parent1:
                child.append(point)
        for point in self.target:
            if point not in child:
                child.append(point)
    
        return child
    
    def mutation(self, population):
        for i in range(len(population)):
            if random.random() <= self.mutation_rate:
                # Seleccionar dos puntos aleatorios en la lista
                index1, index2 = np.random.choice(range(len(population[i])), 2, replace=False)
            
            # Intercambiar los valores en los dos puntos seleccionados
                population[i][index1], population[i][index2] = population[i][index2], population[i][index1]
        return population

    def run_geneticalgo(self): 
        population = self.create_population()

        for i in range(self.n_generations):

            if self.verbose:
                print('_')
                print('Generacion: ', i)
                #print('Poblacion', population)
                print()

            selected = self.selection(population)
            population = self.reproduction(population, selected)
            population = self.mutation(population)

def main():
    target = [
    [24.763531, -107.453989],  # coppel las torres
    [24.747421, -107.426716],  # coppel barrancos
    [24.756552, -107.403092],  # coppel san isidro
    [24.777955, -107.408095],  # coppel clouthier
    [24.739089, -107.393110],  # Coppel canada Clouthier
    [24.777126, -107.432649],  # Coppel bodega Remate
    [24.795033, -107.410673],  # Coppel Plaza fiesta
    [24.797711, -107.423402],  # Coppel la ceiba
    [24.769247, -107.371943],  # Coppel abastos
    [24.798349, -107.393610],  # Coppel Obregon
    [24.806291588486456, -107.39628727881541],  # Coppel Escobedo
    [24.80085076980745, -107.39122225493001],  # Coppel andrade
    [24.816005114020253, -107.42619802671574],  # Coppel Humaya
    [24.808862282043563, -107.39730792212923],  # Coppel Angel flores
    [24.808594949094665, -107.39575474973914],  # Coppel Portales
    [24.81342495074406, -107.40072622765123],  # Coppel plaza Forum
    [24.801446389514645, -107.36533391787977],  # Coppel Madero
    [24.834542368635553, -107.39806426389583],  # Coppel las flores
    [24.8259647322571, -107.42613555285189],  # Coppel sendero
]
    model = DNA(
        target = target,
        mutation_rate = 0.1,
        n_individuals = 3,
        n_selection = 2,
        n_generations = 3,
        verbose=True)
    model.run_geneticalgo()

if __name__ == "__main__":
    main()
