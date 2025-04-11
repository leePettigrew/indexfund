"""
run_all.py

An integrated script to download stock data, compute returns,
and run both an AMPL-based optimization and a GA-based optimization
for constructing a sparse index fund that replicates the S&P 100.

The script provides a menu where the user can choose to:
1. Download historical stock data.
2. Compute daily and quarterly returns.
3. Run AMPL optimization.
4. Run GA optimization.
5. Run all tasks.
0. Exit.

Ensure that:
- The file "sp100_tickers.csv" exists in the ../data folder and contains a "Ticker" column.
- The AMPL model "index_fund.mod" exists in the ../models folder.
- The results will be saved in the ../results folder.
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
import pygad
from amplpy import AMPL


#############################
# Data Download Functionality
#############################
def download_stock_data(tickers, start_date, end_date):
    """
    Downloads the historical adjusted closing prices for a list of tickers
    between start_date and end_date.

    Parameters:
        tickers (list): List of stock ticker symbols.
        start_date (str): Start date (format: 'YYYY-MM-DD').
        end_date (str): End date (format: 'YYYY-MM-DD').

    Returns:
        pd.DataFrame: DataFrame with dates as index and tickers as columns.
    """
    data = {}
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        try:
            ticker_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
            if ticker_data.empty:
                print(f"Warning: No data found for {ticker}. Skipping ticker.")
                continue
            if "Adj Close" not in ticker_data.columns:
                print(f"Warning: 'Adj Close' column not found for {ticker}. Skipping ticker.")
                continue
            # Use squeeze() to ensure it’s a 1D Series.
            adj_close_series = ticker_data["Adj Close"].squeeze()
            if adj_close_series.ndim != 1:
                print(f"Error: Data for {ticker} is not 1-dimensional after squeeze(). Skipping ticker.")
                continue
            data[ticker] = adj_close_series
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
    if not data:
        print("No valid data found for any tickers.")
        return pd.DataFrame()
    df = pd.concat(data, axis=1)
    return df


def run_data_collection():
    """
    Loads tickers from sp100_tickers.csv, downloads historical prices,
    and saves them to sp100_prices.csv.
    """
    current_dir = os.path.dirname(__file__)
    tickers_file = os.path.abspath(os.path.join(current_dir, "..", "data", "sp100_tickers.csv"))
    output_prices_file = os.path.abspath(os.path.join(current_dir, "..", "data", "sp100_prices.csv"))

    print(f"Looking for tickers file at: {tickers_file}")
    if not os.path.exists(tickers_file):
        print(f"Error: Tickers file not found at {tickers_file}. Please create it with a column 'Ticker'.")
        return

    tickers_df = pd.read_csv(tickers_file)
    if "Ticker" not in tickers_df.columns:
        print("Error: CSV file must have a column named 'Ticker'.")
        return

    tickers = tickers_df["Ticker"].dropna().unique().tolist()
    print(f"Found {len(tickers)} tickers.")

    # Define date range for historical data
    start_date = "2023-01-01"
    end_date = "2025-03-01"
    prices_df = download_stock_data(tickers, start_date, end_date)

    if prices_df.empty:
        print("Error: No stock data was downloaded. Exiting.")
        return

    prices_df.to_csv(output_prices_file)
    print(f"Historical price data successfully saved to {output_prices_file}")


#############################
# Return Computation
#############################
def compute_daily_returns(prices_df):
    """
    Compute daily percentage returns from prices.

    Parameters:
        prices_df (pd.DataFrame): DataFrame containing historical prices with datetime index.

    Returns:
        pd.DataFrame: Daily returns.
    """
    return prices_df.pct_change().dropna()


def compute_periodic_returns(daily_returns, period='Q'):
    """
    Compute periodic returns by resampling the daily returns.

    Parameters:
        daily_returns (pd.DataFrame): Daily returns DataFrame.
        period (str): Resampling period ('Q' for quarterly, 'M' for monthly, etc.)

    Returns:
        pd.DataFrame: Periodic returns computed as compounded returns.
    """
    periodic_returns = daily_returns.resample(period).apply(lambda x: (1 + x).prod() - 1)
    return periodic_returns


def run_compute_returns():
    """
    Loads historical prices from sp100_prices.csv, computes daily and quarterly returns,
    and saves them as sp100_daily_returns.csv and sp100_quarterly_returns.csv.
    """
    current_dir = os.path.dirname(__file__)
    prices_file = os.path.abspath(os.path.join(current_dir, "..", "data", "sp100_prices.csv"))
    if not os.path.exists(prices_file):
        print(f"Error: Prices file not found at {prices_file}.")
        return
    prices_df = pd.read_csv(prices_file, index_col=0, parse_dates=True)
    print("Historical prices loaded successfully.")

    daily_returns = compute_daily_returns(prices_df)
    print("Daily returns computed.")

    quarterly_returns = compute_periodic_returns(daily_returns, period='Q')
    print("Quarterly returns computed.")

    daily_returns_file = os.path.abspath(os.path.join(current_dir, "..", "data", "sp100_daily_returns.csv"))
    quarterly_returns_file = os.path.abspath(os.path.join(current_dir, "..", "data", "sp100_quarterly_returns.csv"))

    daily_returns.to_csv(daily_returns_file)
    quarterly_returns.to_csv(quarterly_returns_file)

    print(f"Daily returns saved to: {daily_returns_file}")
    print(f"Quarterly returns saved to: {quarterly_returns_file}")


#############################
# AMPL Optimization Functions
#############################
def prepare_sp100_index_returns(quarterly_returns_df):
    """
    Calculate the S&P 100 index returns based on the equal-weighted average
    of all stocks in the index.
    """
    return quarterly_returns_df.mean(axis=1)


def run_optimization(q_value):
    """
    Runs the AMPL optimization to select q stocks and determine their weights.

    Parameters:
        q_value (int): Number of stocks to select.

    Returns:
        dict or None: Dictionary with selected stocks and tracking error if successful, else None.
    """
    try:
        ampl = AMPL()
        ampl.option["solver"] = "gurobi"

        current_dir = os.path.dirname(__file__)
        model_dir = os.path.join(current_dir, "models")
        model_file = os.path.join(model_dir, "index_fund.mod")
        ampl.read(model_file)

        # Load quarterly returns
        data_file = os.path.join(current_dir, "..", "data", "sp100_quarterly_returns.csv")
        if not os.path.exists(data_file):
            print("Error: sp100_quarterly_returns.csv not found. Please compute returns first.")
            return None

        quarterly_returns_df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        sp100_returns = prepare_sp100_index_returns(quarterly_returns_df)
        n = quarterly_returns_df.shape[1]
        t = quarterly_returns_df.shape[0]

        ampl.param['n'] = n
        ampl.param['q'] = q_value
        ampl.param['T'] = t

        returns_param = ampl.param["returns"]
        for i in range(n):
            for j in range(t):
                returns_param[i + 1, j + 1] = quarterly_returns_df.iloc[j, i]

        sp100_param = ampl.param["sp100_returns"]
        for j in range(t):
            sp100_param[j + 1] = sp100_returns.iloc[j]

        print(f"Solving optimization with q = {q_value}, n = {n}, periods = {t}")
        ampl.solve()

        solve_result = ampl.get_value("solve_result")
        print(f"Solve result: {solve_result}")

        selected_stocks = {}
        try:
            select_var = ampl.get_variable('select')
            weight_var = ampl.get_variable('weight')
            for i in range(1, n + 1):
                if select_var[i].value() > 0.5:
                    stock_ticker = quarterly_returns_df.columns[i - 1]
                    selected_stocks[stock_ticker] = weight_var[i].value()
        except Exception as e:
            print("Error extracting results:", e)

        tracking_error = ampl.obj['tracking_error'].value()

        return {
            'selected_stocks': selected_stocks,
            'tracking_error': tracking_error,
            'num_stocks': len(selected_stocks)
        }
    except Exception as e:
        print("Optimization error:", e)
        return None


#############################
# GA Optimization Functions
#############################
def softmax_function(x):
    """Compute softmax values for a vector x."""
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()


# Global variables for GA optimization (set in run_ga_optimization)
ga_returns_df = None
ga_index_returns = None
ga_q = None


def ga_fitness_func(ga_instance, solution, solution_idx):
    """
    Fitness function for the GA approach.

    Each candidate solution is a vector (length = n) of continuous values.
    The top ga_q indices determine selected stocks and weights (via softmax).
    Fitness is defined as the inverse of the tracking error.
    """
    global ga_returns_df, ga_index_returns, ga_q

    selected_idx = np.argsort(solution)[-ga_q:]
    selected_genes = solution[selected_idx]
    weights = softmax_function(selected_genes)
    portfolio_returns = np.dot(ga_returns_df.iloc[:, selected_idx].values, weights)
    te = np.sum((portfolio_returns - ga_index_returns.values) ** 2)
    fitness = 1.0 / (te + 1e-8)
    return fitness


def run_ga_optimization(q_value):
    """
    Runs the GA optimization to select q stocks and determine their weights.

    Parameters:
        q_value (int): Number of stocks to select.
    """
    global ga_returns_df, ga_index_returns, ga_q

    current_dir = os.path.dirname(__file__)
    data_file = os.path.join(current_dir, "..", "data", "sp100_quarterly_returns.csv")
    if not os.path.exists(data_file):
        print("Error: sp100_quarterly_returns.csv not found. Please compute returns first.")
        return

    ga_returns_df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    ga_index_returns = ga_returns_df.mean(axis=1)
    ga_q = q_value

    num_generations = 100
    sol_per_pop = 50
    num_genes = ga_returns_df.shape[1]
    gene_space = {'low': -10, 'high': 10}
    num_parents_mating = sol_per_pop // 2

    ga_instance = pygad.GA(
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_parents_mating=num_parents_mating,
        num_genes=num_genes,
        fitness_func=ga_fitness_func,
        mutation_percent_genes=10,
        mutation_type="random",
        crossover_type="single_point",
        gene_space=gene_space,
        random_seed=42,
        on_generation=lambda ga: print(
            f"GA Generation: {ga.generations_completed}, Best Fitness: {ga.best_solution()[1]}")
    )

    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"\nGA Best solution fitness: {solution_fitness}")

    selected_idx = np.argsort(solution)[-ga_q:]
    selected_tickers = ga_returns_df.columns[selected_idx].tolist()
    selected_genes = solution[selected_idx]
    weights = softmax_function(selected_genes)
    portfolio_returns = np.dot(ga_returns_df.iloc[:, selected_idx].values, weights)
    tracking_error = np.sum((portfolio_returns - ga_index_returns.values) ** 2)

    print(f"GA Tracking Error: {tracking_error:.8f}")
    print("Selected stocks and weights:")
    for ticker, w in zip(selected_tickers, weights):
        print(f"{ticker}: {w:.4f}")

    output_dir = os.path.join(current_dir, "..", "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"ga_optimization_q{ga_q}.csv")
    result_df = pd.DataFrame({"Ticker": selected_tickers, "Weight": weights})
    result_df.to_csv(output_file, index=False)
    print(f"GA results saved to {output_file}")


#############################
# Main Menu
#############################
def main_menu():
    while True:
        print("\nMain Menu:")
        print("1. Download stock data")
        print("2. Compute returns (daily and quarterly)")
        print("3. Run AMPL optimization")
        print("4. Run GA optimization")
        print("5. Run all tasks")
        print("0. Exit")
        choice = input("Please enter your choice: ")

        if choice == "1":
            run_data_collection()
        elif choice == "2":
            run_compute_returns()
        elif choice == "3":
            try:
                q_val = int(input("Enter number of stocks to select (q) for AMPL optimization: "))
            except Exception as e:
                print("Invalid input:", e)
                continue
            result = run_optimization(q_val)
            if result is not None:
                print("AMPL Optimization Results:")
                print(result)
        elif choice == "4":
            try:
                q_val = int(input("Enter number of stocks to select (q) for GA optimization: "))
            except Exception as e:
                print("Invalid input:", e)
                continue
            run_ga_optimization(q_val)
        elif choice == "5":
            print("Running all tasks in sequence:")
            run_data_collection()
            run_compute_returns()
            try:
                q_val = int(input("Enter number of stocks to select (q) for AMPL optimization: "))
            except Exception as e:
                print("Invalid input:", e)
                continue
            result = run_optimization(q_val)
            if result is not None:
                print("AMPL Optimization Results:")
                print(result)
            try:
                q_val_ga = int(input("Enter number of stocks to select (q) for GA optimization: "))
            except Exception as e:
                print("Invalid input:", e)
                continue
            run_ga_optimization(q_val_ga)
        elif choice == "0":
            print("Exiting.")
            break
        else:
            print("Invalid choice, please try again.")


if __name__ == "__main__":
    main_menu()
