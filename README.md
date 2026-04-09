# TSP-GA-Optimization

**Solving the Traveling Salesman Problem using a Genetic Algorithm**  
**PUSL2100 – Evolutionary Computing | Coursework 1 (C1)**

## Overview

This project implements a **Genetic Algorithm (GA)** to solve a 100-city instance of the **Traveling Salesman Problem (TSP)**.  
The objective is to find the shortest possible route that visits each city exactly once and returns to the starting city.

The solution was developed as group coursework for the module **PUSL2100 Evolutionary Computing** at the University of Plymouth (NSBM Green University).

**Best distance achieved:** `793` (replace with your actual best distance from the final run)

---

## Features Implemented

- **Representation**: Permutation encoding (chromosome is a list of cities 0–99 with no duplicates)
- **Fitness Function**: Total tour distance (accelerated with Numba JIT)
- **Selection**: Tournament Selection (size = 5)
- **Crossover**:
  - Order Crossover (OX) – primary operator
  - Partially Mapped Crossover (PMX) – implemented as alternative
- **Mutation**: Swap Mutation
- **Local Search**: Single-pass 2-opt heuristic (applied selectively)
- **Elitism**: Best individual preserved every generation
- **Enhancements**: Larger population (200), more generations (800), and tuned parameters for improved convergence

---

## Requirements & Dependencies

### Required Files
- `tsp_data_100_distance_matrix.csv` (already included in the repository)

### Python Dependencies
```bash
pip install pandas numpy matplotlib tqdm numba
```
### Used libraries:

- pandas – loading the distance matrix
- numpy – array operations and random permutations
- matplotlib – generating convergence plots
- tqdm – progress bar during evolution
- numba – fast JIT-compiled fitness evaluation

How to Run

Clone the repository:Bashgit clone https://github.com/ChaXRium/TSP-GA-Optimization.git
cd TSP-GA-Optimization
Install dependencies:Bashpip install pandas numpy matplotlib tqdm numba
Run the Genetic Algorithm:Bashpython tsp_ga.py

The script will:

Evolve the population for 800 generations
Print progress every 50 generations
Output the best tour found and its total distance
Save a convergence plot (convergence_plot_OX_2opt.png)
