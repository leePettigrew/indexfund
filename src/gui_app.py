"""
gui_app.py

A graphical interface to:
  1. Download stock data from Yahoo Finance.
  2. Compute daily and quarterly returns.
  3. Run an AMPL optimization for constructing a sparse S&P 100 index.
  4. Run a Genetic Algorithm (GA) optimization for the same purpose.
  5. Run all tasks in sequence.

The GUI displays logs and, when an optimization is run, shows a bar chart
of selected stocks and weights.

Ensure required files and folders:
- sp100_tickers.csv in ../data with a "Ticker" column.
- index_fund.mod in ../models.
- Results will be saved in ../results.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, scrolledtext
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pygad
from amplpy import AMPL


# -------------------------------
# Helper Functions
# -------------------------------
def softmax(x):
    ex = np.exp(x - np.max(x))  # for numerical stability
    return ex / ex.sum()


# -------------------------------
# Data Download Functions
# -------------------------------
def download_stock_data(tickers, start_date, end_date, log_func):
    log_func("Downloading stock data...")
    data = {}
    for ticker in tickers:
        log_func(f"Downloading data for {ticker}...")
        try:
            ticker_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
            if ticker_data.empty:
                log_func(f"Warning: No data for {ticker}.")
                continue
            if "Adj Close" not in ticker_data.columns:
                log_func(f"Warning: 'Adj Close' not found for {ticker}.")
                continue
            adj_close = ticker_data["Adj Close"].squeeze()
            if adj_close.ndim != 1:
                log_func(f"Error: Data for {ticker} is not 1D.")
                continue
            data[ticker] = adj_close
        except Exception as e:
            log_func(f"Error downloading {ticker}: {e}")
    if not data:
        log_func("No valid data downloaded.")
        return pd.DataFrame()
    df = pd.concat(data, axis=1)
    log_func("Downloaded data for stocks.")
    return df


def run_data_collection(log_func):
    current_dir = os.path.dirname(__file__)
    tickers_file = os.path.abspath(os.path.join(current_dir, "..", "data", "sp100_tickers.csv"))
    output_prices_file = os.path.abspath(os.path.join(current_dir, "..", "data", "sp100_prices.csv"))

    log_func(f"Looking for tickers file at: {tickers_file}")
    if not os.path.exists(tickers_file):
        log_func(f"Error: Tickers file not found at {tickers_file}.")
        return
    tickers_df = pd.read_csv(tickers_file)
    if "Ticker" not in tickers_df.columns:
        log_func("Error: CSV must have column 'Ticker'.")
        return
    tickers = tickers_df["Ticker"].dropna().unique().tolist()
    log_func(f"Found {len(tickers)} tickers.")

    start_date = "2023-01-01"
    end_date = "2025-03-01"
    prices_df = download_stock_data(tickers, start_date, end_date, log_func)
    if prices_df.empty:
        log_func("Error: No stock data downloaded.")
        return
    prices_df.to_csv(output_prices_file)
    log_func(f"Historical prices saved to {output_prices_file}")


# -------------------------------
# Returns Computation Functions
# -------------------------------
def compute_daily_returns(prices_df):
    return prices_df.pct_change().dropna()


def compute_periodic_returns(daily_returns, period='Q'):
    periodic_returns = daily_returns.resample(period).apply(lambda x: (1 + x).prod() - 1)
    return periodic_returns


def run_compute_returns(log_func):
    current_dir = os.path.dirname(__file__)
    prices_file = os.path.abspath(os.path.join(current_dir, "..", "data", "sp100_prices.csv"))
    if not os.path.exists(prices_file):
        log_func(f"Error: Prices file not found at {prices_file}.")
        return
    prices_df = pd.read_csv(prices_file, index_col=0, parse_dates=True)
    log_func("Historical prices loaded.")

    daily_returns = compute_daily_returns(prices_df)
    log_func("Daily returns computed.")

    quarterly_returns = compute_periodic_returns(daily_returns, period='Q')
    log_func("Quarterly returns computed.")

    daily_returns_file = os.path.abspath(os.path.join(current_dir, "..", "data", "sp100_daily_returns.csv"))
    quarterly_returns_file = os.path.abspath(os.path.join(current_dir, "..", "data", "sp100_quarterly_returns.csv"))
    daily_returns.to_csv(daily_returns_file)
    quarterly_returns.to_csv(quarterly_returns_file)

    log_func(f"Daily returns saved to: {daily_returns_file}")
    log_func(f"Quarterly returns saved to: {quarterly_returns_file}")


# -------------------------------
# AMPL Optimization Functions
# -------------------------------
def prepare_sp100_index_returns(quarterly_returns_df):
    return quarterly_returns_df.mean(axis=1)


def run_optimization(q_value, log_func):
    try:
        ampl = AMPL()
        ampl.option["solver"] = "gurobi"
        current_dir = os.path.dirname(__file__)
        model_dir = os.path.join(current_dir, "models")
        model_file = os.path.join(model_dir, "index_fund.mod")
        ampl.read(model_file)

        data_file = os.path.join(current_dir, "..", "data", "sp100_quarterly_returns.csv")
        if not os.path.exists(data_file):
            log_func("Error: sp100_quarterly_returns.csv not found. Compute returns first.")
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

        log_func(f"Solving AMPL optimization with q = {q_value}, n = {n}, periods = {t}...")
        ampl.solve()
        solve_result = ampl.get_value("solve_result")
        log_func(f"Solve result: {solve_result}")

        selected_stocks = {}
        try:
            select_var = ampl.get_variable('select')
            weight_var = ampl.get_variable('weight')
            for i in range(1, n + 1):
                if select_var[i].value() > 0.5:
                    stock_ticker = quarterly_returns_df.columns[i - 1]
                    selected_stocks[stock_ticker] = weight_var[i].value()
        except Exception as e:
            log_func(f"Error extracting results: {e}")

        tracking_error = ampl.obj['tracking_error'].value()
        result = {
            'selected_stocks': selected_stocks,
            'tracking_error': tracking_error,
            'num_stocks': len(selected_stocks)
        }
        log_func(f"AMPL Optimization complete. Tracking Error: {tracking_error:.8e}")
        return result
    except Exception as e:
        log_func(f"AMPL Optimization error: {e}")
        return None


# -------------------------------
# GA Optimization Functions
# -------------------------------
def ga_fitness_func(ga_instance, solution, solution_idx):
    global ga_returns_df, ga_index_returns, ga_q
    selected_idx = np.argsort(solution)[-ga_q:]
    selected_genes = solution[selected_idx]
    weights = softmax(selected_genes)
    portfolio_returns = np.dot(ga_returns_df.iloc[:, selected_idx].values, weights)
    te = np.sum((portfolio_returns - ga_index_returns.values) ** 2)
    fitness = 1.0 / (te + 1e-8)
    return fitness


# Global variables for GA optimization
ga_returns_df = None
ga_index_returns = None
ga_q = None


def run_ga_optimization(q_value, log_func):
    global ga_returns_df, ga_index_returns, ga_q
    current_dir = os.path.dirname(__file__)
    data_file = os.path.join(current_dir, "..", "data", "sp100_quarterly_returns.csv")
    if not os.path.exists(data_file):
        log_func("Error: sp100_quarterly_returns.csv not found. Compute returns first.")
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
        on_generation=lambda ga: log_func(
            f"GA Generation {ga.generations_completed}, Best Fitness: {ga.best_solution()[1]}")
    )

    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    log_func(f"GA Best solution fitness: {solution_fitness}")

    selected_idx = np.argsort(solution)[-ga_q:]
    selected_tickers = ga_returns_df.columns[selected_idx].tolist()
    selected_genes = solution[selected_idx]
    weights = softmax(selected_genes)
    portfolio_returns = np.dot(ga_returns_df.iloc[:, selected_idx].values, weights)
    tracking_error = np.sum((portfolio_returns - ga_index_returns.values) ** 2)

    log_func(f"GA Tracking Error: {tracking_error:.8e}")
    log_func("GA Selected stocks and weights:")
    for ticker, w in zip(selected_tickers, weights):
        log_func(f"  {ticker}: {w:.4f}")

    output_dir = os.path.join(current_dir, "..", "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"ga_optimization_q{ga_q}.csv")
    result_df = pd.DataFrame({"Ticker": selected_tickers, "Weight": weights})
    result_df.to_csv(output_file, index=False)
    log_func(f"GA results saved to {output_file}")

    # Plot a bar chart for visual aid
    plot_optimization_results(selected_tickers, weights, title=f"GA Optimization (q={ga_q})")


# -------------------------------
# Visualization Function
# -------------------------------
def plot_optimization_results(tickers, weights, title="Optimization Results"):
    plt.figure(figsize=(10, 6))
    plt.bar(tickers, weights)
    plt.xlabel("Stock Ticker")
    plt.ylabel("Weight")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# -------------------------------
# GUI Class
# -------------------------------
class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Index Fund Construction via AI-driven Methods")
        self.create_widgets()

    def create_widgets(self):
        # Frame for buttons
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.grid(row=0, column=0, sticky="ew")

        ttk.Button(button_frame, text="Download Stock Data", command=self.download_data).grid(row=0, column=0, padx=5,
                                                                                              pady=5)
        ttk.Button(button_frame, text="Compute Returns", command=self.compute_returns).grid(row=0, column=1, padx=5,
                                                                                            pady=5)
        ttk.Button(button_frame, text="Run AMPL Optimization", command=self.run_ampl_opt).grid(row=0, column=2, padx=5,
                                                                                               pady=5)
        ttk.Button(button_frame, text="Run GA Optimization", command=self.run_ga_opt).grid(row=0, column=3, padx=5,
                                                                                           pady=5)
        ttk.Button(button_frame, text="Run All Tasks", command=self.run_all_tasks).grid(row=0, column=4, padx=5, pady=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).grid(row=0, column=5, padx=5, pady=5)

        # Scrolled text area for logs
        self.log_text = scrolledtext.ScrolledText(self.root, width=100, height=20)
        self.log_text.grid(row=1, column=0, padx=10, pady=10)

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def download_data(self):
        self.log("Starting data download...")
        run_data_collection(self.log)
        self.log("Data download complete.")

    def compute_returns(self):
        self.log("Starting computation of returns...")
        run_compute_returns(self.log)
        self.log("Return computation complete.")

    def run_ampl_opt(self):
        try:
            q_val = simpledialog.askinteger("AMPL Optimization", "Enter number of stocks to select (q):", minvalue=1)
            if q_val is None:
                return
            self.log(f"Running AMPL Optimization with q = {q_val} ...")
            result = run_optimization(q_val, self.log)
            if result is not None:
                self.log("AMPL Optimization Results:")
                self.log(str(result))
                # Optionally, you could add a visualization if desired.
            else:
                self.log("AMPL Optimization did not produce results.")
        except Exception as e:
            messagebox.showerror("Error", f"Error in AMPL Optimization: {e}")

    def run_ga_opt(self):
        try:
            q_val = simpledialog.askinteger("GA Optimization", "Enter number of stocks to select (q):", minvalue=1)
            if q_val is None:
                return
            self.log(f"Running GA Optimization with q = {q_val} ...")
            run_ga_optimization(q_val, self.log)
        except Exception as e:
            messagebox.showerror("Error", f"Error in GA Optimization: {e}")

    def run_all_tasks(self):
        self.download_data()
        self.compute_returns()
        try:
            q_val = simpledialog.askinteger("AMPL Optimization", "Enter number of stocks to select (q) for AMPL:",
                                            minvalue=1)
            if q_val is None:
                return
            self.log(f"Running AMPL Optimization with q = {q_val} ...")
            result = run_optimization(q_val, self.log)
            if result is not None:
                self.log("AMPL Optimization Results:")
                self.log(str(result))
            q_val_ga = simpledialog.askinteger("GA Optimization", "Enter number of stocks to select (q) for GA:",
                                               minvalue=1)
            if q_val_ga is None:
                return
            self.log(f"Running GA Optimization with q = {q_val_ga} ...")
            run_ga_optimization(q_val_ga, self.log)
        except Exception as e:
            messagebox.showerror("Error", f"Error running all tasks: {e}")


# -------------------------------
# Main
# -------------------------------
def main():
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
