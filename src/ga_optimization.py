"""
ga_optimization.py

An implementation of a genetic algorithm (GA) for sparse portfolio selection
that replicates the S&P 100 index returns using historical quarterly returns.
The GA selects a subset (q stocks) from all available stocks and assigns weights
to minimize the tracking error between the portfolio and the equal-weighted index.

Before running:
- Ensure you have PyGAD installed: pip install pygad
- Make sure the data file "sp100_quarterly_returns.csv" exists in the ../data/ folder relative to this file.
- The GA output will be saved to the ../results/ folder.
"""

import os
import numpy as np
import pandas as pd
import pygad


# -------------------------------
# Helper function: softmax
# -------------------------------
def softmax(x):
    """Compute softmax values for a vector x."""
    ex = np.exp(x - np.max(x))  # subtract max(x) for numerical stability
    return ex / ex.sum()


# -------------------------------
# Global variables (to be set in main)
# -------------------------------
returns_df = None       # DataFrame of quarterly returns, shape (T, n)
index_returns = None    # Series for equal-weighted S&P100 returns, shape (T,)
q = None                # Number of stocks to select
n = None                # Total number of stocks


# -------------------------------
# Fitness function for GA
# -------------------------------
def fitness_func(ga_instance, solution, solution_idx):
    """
    Evaluates a candidate solution.

    Each candidate 'solution' is a vector (length = n) of continuous values.

    Process:
      - The top q indices (largest gene values) indicate the selected stocks.
      - The weights for the selected stocks are computed using softmax on these gene values.
      - The portfolio return for each period is computed as the dot-product between
        the returns for the selected stocks and the computed weights.
      - The tracking error is the sum of squared differences between the portfolio returns and the benchmark index.

    The fitness score is defined as 1 / (tracking error + epsilon), so a lower tracking error gives a higher fitness.
    """
    global returns_df, index_returns, q

    # Get indices of the q stocks with the highest gene values
    selected_idx = np.argsort(solution)[-q:]

    # Compute weights using softmax on the selected gene values
    selected_genes = solution[selected_idx]
    weights = softmax(selected_genes)

    # Calculate portfolio returns for each period for the selected stocks
    portfolio_returns = np.dot(returns_df.iloc[:, selected_idx].values, weights)

    # Compute tracking error: sum of squared differences between portfolio and index returns
    te = np.sum((portfolio_returns - index_returns.values) ** 2)

    # Return fitness = 1 / (tracking error + epsilon) to avoid division by zero
    fitness = 1.0 / (te + 1e-8)
    return fitness


# -------------------------------
# Optional: Callback function to monitor progress
# -------------------------------
def on_generation(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}, Best Fitness = {ga_instance.best_solution()[1]}")


# -------------------------------
# Main function: Run the GA-based portfolio selection
# -------------------------------
def main():
    global returns_df, index_returns, q, n

    # --- Load the quarterly returns data ---
    current_dir = os.path.dirname(__file__)
    data_file = os.path.join(current_dir, "..", "data", "sp100_quarterly_returns.csv")

    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}.")
        return

    # Load data (CSV should have dates as the index and stocks as columns)
    returns_df = pd.read_csv(data_file, index_col=0, parse_dates=True)

    # Get the number of periods (rows) and stocks (columns)
    T, n = returns_df.shape
    print(f"Loaded quarterly returns: {T} periods, {n} stocks.")

    # --- Compute the equal-weighted S&P100 index returns ---
    index_returns = returns_df.mean(axis=1)  # Series of length T

    # --- Set the number of stocks to be selected (q) ---
    q = 20  # Adjust this value as needed (e.g., 10, 15, 20, 25, 30)

    # --- GA Settings ---
    num_generations = 100     # Total number of generations
    sol_per_pop = 50          # Population size (solutions per generation)
    num_genes = n             # Number of genes in each solution (one per stock)

    # Define gene space: allowable range for each gene
    gene_space = {'low': -10, 'high': 10}

    # num_parents_mating is a required argument for mating.
    num_parents_mating = sol_per_pop // 2  # Use half of the population for mating

    # --- Create the GA instance ---
    ga_instance = pygad.GA(
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_parents_mating=num_parents_mating,
        num_genes=num_genes,
        fitness_func=fitness_func,
        mutation_percent_genes=10,
        mutation_type="random",
        crossover_type="single_point",
        on_generation=on_generation,
        gene_space=gene_space,
        random_seed=42  # Fixed seed for reproducibility
    )

    # --- Run the GA optimization ---
    ga_instance.run()

    # --- Retrieve and interpret the best solution ---
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"\nBest solution fitness: {solution_fitness}")

    # Determine selected stocks and compute their weights using softmax
    selected_idx = np.argsort(solution)[-q:]
    selected_tickers = returns_df.columns[selected_idx].tolist()
    selected_genes = solution[selected_idx]
    weights = softmax(selected_genes)

    # --- Compute tracking error for the best solution ---
    portfolio_returns = np.dot(returns_df.iloc[:, selected_idx].values, weights)
    tracking_error = np.sum((portfolio_returns - index_returns.values) ** 2)
    print(f"Tracking Error: {tracking_error:.8f}")
    print(f"Number of stocks selected: {len(selected_tickers)}")
    print("Selected stocks and weights:")
    for ticker, w in zip(selected_tickers, weights):
        print(f"{ticker}: {w:.4f}")

    # --- Save results to CSV ---
    output_dir = os.path.join(current_dir, "..", "results")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"ga_optimization_q{q}.csv")
    result_df = pd.DataFrame({
        "Ticker": selected_tickers,
        "Weight": weights
    })
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
