import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from numba import njit
from tqdm import tqdm

def load_distance_matrix(filename):
    df = pd.read_csv(filename, index_col=0)
    return df.values.astype(float)

class GeneticAlgorithmTSP:
    def __init__(self, dist_matrix, pop_size=200, generations=800,
                 tournament_size=5, crossover_rate=0.8, mutation_rate=0.2,
                 local_search_prob=0.8, crossover_type='OX'):
        self.dist_matrix = dist_matrix
        self.n = dist_matrix.shape[0]
        self.pop_size = pop_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.local_search_prob = local_search_prob
        self.crossover_type = crossover_type.upper()

        # Hall of Fame (Global Best)
        self.hall_of_fame = None
        self.hall_of_fame_distance = float('inf')

    def create_individual(self):
        return list(np.random.permutation(self.n))

    def create_population(self):
        return [self.create_individual() for _ in range(self.pop_size)]

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

    def order_crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [-1] * size
        child[start:end] = parent1[start:end]
        pos = end % size
        for gene in parent2:
            if gene not in child:
                while child[pos] != -1:
                    pos = (pos + 1) % size
                child[pos] = gene
                pos = (pos + 1) % size
        return child

    def pmx_crossover(self, parent1, parent2):
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

    def swap_mutation(self, individual):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(self.n), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    def two_opt_local_search(self, individual):
        for i in range(self.n - 1):
            for j in range(i + 2, self.n):
                d1 = (self.dist_matrix[individual[i]][individual[i + 1]] +
                      self.dist_matrix[individual[j]][individual[(j + 1) % self.n]])
                d2 = (self.dist_matrix[individual[i]][individual[j]] +
                      self.dist_matrix[individual[i + 1]][individual[(j + 1) % self.n]])
                if d2 < d1:
                    individual[i + 1:j + 1] = individual[i + 1:j + 1][::-1]
        return individual

    def evolve(self):
        population = self.create_population()
        best_distances = []
        start_time = time.time()

        print(f"Using {self.crossover_type} + 2-opt (prob={self.local_search_prob}) + Optimized Hall of Fame")
        print(f"Pop={self.pop_size} | Gen={self.generations}")

        for gen in tqdm(range(self.generations), desc="Evolving GA", leave=True):
            fitnesses = [self.calculate_fitness(ind) for ind in population]
            best_dist = min(fitnesses)
            best_distances.append(best_dist)

            # Update Hall of Fame
            if best_dist < self.hall_of_fame_distance:
                best_idx = np.argmin(fitnesses)
                self.hall_of_fame = population[best_idx][:]
                self.hall_of_fame_distance = best_dist

            if gen % 50 == 0 or gen == self.generations - 1:
                elapsed = time.time() - start_time
                print(f" → Best: {best_dist:8.2f} | HoF: {self.hall_of_fame_distance:8.2f} | Time: {elapsed:.1f}s")

            selected = self.tournament_selection(population, fitnesses)

            next_pop = []
            for i in range(0, self.pop_size, 2):
                if random.random() < self.crossover_rate and i + 1 < self.pop_size:
                    if self.crossover_type == 'PMX':
                        child1 = self.pmx_crossover(selected[i], selected[i + 1])
                        child2 = self.pmx_crossover(selected[i + 1], selected[i])
                    else:
                        child1 = self.order_crossover(selected[i], selected[i + 1])
                        child2 = self.order_crossover(selected[i + 1], selected[i])
                    next_pop.extend([child1, child2])
                else:
                    next_pop.extend([selected[i][:], selected[i + 1][:] if i + 1 < self.pop_size else selected[i][:]])

            # Mutation + 2-opt (only on first 20)
            for idx, ind in enumerate(next_pop):
                self.swap_mutation(ind)
                if random.random() < self.local_search_prob and idx < 20:
                    self.two_opt_local_search(ind)

            # === Optimized Hall of Fame Elitism ===
            if self.hall_of_fame is not None:
                # Always keep the current best in position 0 (light elitism)
                best_idx = np.argmin(fitnesses)
                next_pop[0] = population[best_idx][:]

                # Replace the worst in next_pop with Hall of Fame (but not too often)
                if gen % 5 == 0 or best_dist > self.hall_of_fame_distance * 1.02:   # insert more when stuck
                    next_fitnesses = [self.calculate_fitness(ind) for ind in next_pop]
                    worst_idx = np.argmax(next_fitnesses)
                    next_pop[worst_idx] = self.hall_of_fame[:]

            population = next_pop

        print(f"\n Evolution finished in {time.time() - start_time:.1f} seconds")

        # Return the Hall of Fame as final best
        best_tour = [int(city) for city in self.hall_of_fame]
        best_distance = self.hall_of_fame_distance

        print(f"Final Best Distance (Hall of Fame): {best_distance:.2f}")

        return best_tour, best_distance, best_distances


if __name__ == "__main__":
    print("Loading distance matrix...")
    dist_matrix = load_distance_matrix("tsp_data_100_distance_matrix.csv")

    ga = GeneticAlgorithmTSP(dist_matrix, 
                             pop_size=200, 
                             generations=800,
                             crossover_type='OX',
                             local_search_prob=0.8)

    print("Starting GA evolution...")
    best_tour, best_distance, convergence = ga.evolve()

    print("\n" + "="*70)
    print(" BEST SOLUTION FOUND (Hall of Fame)")
    print("="*70)
    print("Tour (city order):")
    print(best_tour)
    print(f"Total distance      : {best_distance:.2f}")
    print("="*70)

    # Plot
    plot_name = f"convergence_plot_{ga.crossover_type}_hof.png"
    plt.figure(figsize=(10, 5))
    plt.plot(convergence)
    plt.title(f"GA Convergence - TSP 100 Cities (with Hall of Fame)")
    plt.xlabel("Generation")
    plt.ylabel("Best Tour Distance")
    plt.grid(True)
    plt.savefig(plot_name)
    plt.show()
