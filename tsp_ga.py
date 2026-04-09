import pandas as pd
import numpy as np
import random
import time
from tqdm import tqdm
from numba import njit
import matplotlib.pyplot as plt

def load_distance_matrix(filename):
    df = pd.read_csv(filename, index_col=0)
    return df.values.astype(float)

class GeneticAlgorithmTSP:
    def __init__(self, dist_matrix, pop_size=100, generations=500,
                 tournament_size=5, crossover_rate=0.8, mutation_rate=0.2,
                 local_search_prob=0.1, crossover_type='OX'):
        self.dist_matrix = dist_matrix
        self.n = dist_matrix.shape[0]
        self.pop_size = pop_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.local_search_prob = local_search_prob
        self.crossover_type = crossover_type.upper()

    def create_individual(self):
        return list(np.random.permutation(self.n))

    def create_population(self):
        return [self.create_individual() for _ in range(self.pop_size)]

    def order_crossover(self, parent1, parent2):
        """Order Crossover (OX)"""
        size = len(parent1)
        child = [-1] * size
        start, end = sorted(random.sample(range(size), 2))
        child[start:end] = parent1[start:end]

        pos = end % size
        for gene in parent2[end:] + parent2[:end]:
            if gene not in child:
                child[pos] = gene
                pos = (pos + 1) % size
        return child

    def pmx_crossover(self, parent1, parent2):
        """Partially Mapped Crossover (PMX)"""
        size = len(parent1)
        child = [-1] * size
        start, end = sorted(random.sample(range(size), 2))
        child[start:end] = parent1[start:end]

        mapping = {}
        for i in range(start, end):
            mapping[parent2[i]] = parent1[i]

        pos = 0
        for gene in parent2:
            if gene not in child:
                while child[pos] != -1:
                    pos = (pos + 1) % size
                if gene in mapping:
                    mapped_gene = mapping[gene]
                    while mapped_gene in child:
                        mapped_gene = mapping.get(mapped_gene, mapped_gene)
                    child[pos] = mapped_gene
                else:
                    child[pos] = gene
                pos = (pos + 1) % size
        return child

    @staticmethod
    @njit(fastmath=True)
    def calculate_fitness_numba(dist_matrix, individual):
        n = len(individual)
        distance = 0.0
        for i in range(n):
            distance += dist_matrix[individual[i]][individual[(i + 1) % n]]
        return distance

    def calculate_fitness(self, individual):
        return GeneticAlgorithmTSP.calculate_fitness_numba(self.dist_matrix, np.array(individual))

    def tournament_selection(self, population, fitnesses):
        selected = []
        for _ in range(self.pop_size):
            competitors = random.sample(range(len(population)), self.tournament_size)
            winner = min(competitors, key=lambda i: fitnesses[i])
            selected.append(population[winner][:])
        return selected

    def evolve(self):
        population = self.create_population()
        best_distances = []
        start_time = time.time()

        for gen in tqdm(range(self.generations), desc="Evolving"):
            fitnesses = [self.calculate_fitness(ind) for ind in population]
            best_dist = min(fitnesses)
            best_distances.append(best_dist)

            selected = self.tournament_selection(population, fitnesses)

            next_pop = []
            for i in range(0, self.pop_size, 2):
                if random.random() < self.crossover_rate and i + 1 < self.pop_size:
                    parent1 = selected[i]
                    parent2 = selected[i + 1]
                    if self.crossover_type == 'PMX':
                        child1 = self.pmx_crossover(parent1, parent2)
                        child2 = self.pmx_crossover(parent2, parent1)
                    else:
                        child1 = self.order_crossover(parent1, parent2)
                        child2 = self.order_crossover(parent2, parent1)
                    next_pop.extend([child1, child2])
                else:
                    next_pop.extend([selected[i][:], selected[i + 1][:]])

            population = next_pop

        print(f"\n✅ Evolution finished in {time.time() - start_time:.1f} seconds")
        final_fitnesses = [self.calculate_fitness(ind) for ind in population]
        best_idx = np.argmin(final_fitnesses)
        best_tour = [int(city) for city in population[best_idx]]
        best_distance = final_fitnesses[best_idx]

        # Convergence plot
        plt.figure(figsize=(10, 5))
        plt.plot(best_distances)
        plt.title(f"GA Convergence - TSP 100 Cities ({self.crossover_type} + 2-opt)")
        plt.xlabel("Generation")
        plt.ylabel("Best Tour Distance")
        plt.grid(True)
        plt.savefig(f"convergence_plot_{self.crossover_type}_2opt.png")
        plt.show()

        return best_tour, best_distance, best_distances


if __name__ == "__main__":
    dist_matrix = load_distance_matrix("tsp_data_100_distance_matrix.csv")
    ga = GeneticAlgorithmTSP(dist_matrix)
    best_tour, best_distance, _ = ga.evolve()
    print("Best distance:", best_distance)