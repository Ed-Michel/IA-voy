import random
from deap import creator, base, tools, algorithms

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    return sum(individual),

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=300)

NGEN=40
for gen in range(NGEN):
    print(f"Generation {gen + 1}/{NGEN}")
    
    # Variation
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    
    # Evaluate the fitness of the offspring
    fits = toolbox.map(toolbox.evaluate, offspring)
    
    # Assign the fitness values to the offspring individuals
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    
    # Select individuals for the next generation
    population = toolbox.select(offspring, k=len(population))
    
    # Print the top individual of the current generation
    top_ind = tools.selBest(population, k=1)[0]
    print(f"Top individual of generation {gen + 1}: {top_ind}, Fitness: {top_ind.fitness.values[0]}")

# Select the top 10 individuals from the final population
top10 = tools.selBest(population, k=10)
print("\nTop 10 individuals:")
for i, ind in enumerate(top10, 1):
    print(f"#{i}: {ind}, Fitness: {ind.fitness.values[0]}")
