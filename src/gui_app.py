"""
Enhanced Index Fund Constructor (EIFC) - AI-Driven Portfolio Optimization Application

A comprehensive application for creating optimal index fund portfolios that track the S&P 100
using fewer stocks. This application integrates data collection, multiple optimization methods,
advanced visualizations, and comparative analysis.

Features:
1. Data Management:
   - Download S&P 100 stock data with progress tracking
   - Compute daily and quarterly returns with data integrity checks
   - Data visualization and statistics

2. Portfolio Optimization:
   - AMPL optimization with Gurobi solver
   - Genetic Algorithm optimization
   - Parameter tuning interface
   - Risk tolerance adjustments

3. Analysis & Visualization:
   - Interactive portfolio composition charts
   - Performance comparison with S&P 100 benchmark
   - Efficient frontier visualization
   - Optimization convergence tracking
   - Backtest simulation

4. Export & Reporting:
   - Save results in multiple formats
   - Generate performance reports
   - Compare optimization methods

Requirements:
- customtkinter, matplotlib, pandas, numpy, yfinance, pygad, amplpy

Created: April 12, 2025
"""

import os
import sys
import time
import datetime
import threading
import warnings
import logging
from typing import Dict, List, Tuple, Union, Optional, Callable

# Data handling
import pandas as pd
import numpy as np
import yfinance as yf

# Optimization
import pygad
from amplpy import AMPL

# Visualization
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from matplotlib.cm import get_cmap

# GUI
import customtkinter as ctk
from tkinter import filedialog, messagebox, StringVar, BooleanVar, DoubleVar, IntVar

# Set appearance and styling
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
plt.style.use("dark_background")
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Helper functions for optimization
def softmax(x):
    """Compute softmax values for a vector x."""
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()

def calculate_tracking_error(portfolio_returns, benchmark_returns):
    """Calculate tracking error between portfolio and benchmark returns."""
    return np.sqrt(np.mean((portfolio_returns - benchmark_returns) ** 2))

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate the Sharpe ratio for a series of returns."""
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0

class DataManager:
    """Handles data collection, processing, and management for the application."""

    def __init__(self):
        """Initialize the DataManager with default paths and settings."""
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the project root directory
        self.project_root = os.path.abspath(os.path.join(self.current_dir, ".."))
        # Data and results directories are at the project root level
        self.data_dir = os.path.join(self.project_root, "data")
        self.results_dir = os.path.join(self.project_root, "results")

        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # File paths
        self.tickers_file = os.path.join(self.data_dir, "sp100_tickers.csv")
        self.prices_file = os.path.join(self.data_dir, "sp100_prices.csv")
        self.daily_returns_file = os.path.join(self.data_dir, "sp100_daily_returns.csv")
        self.quarterly_returns_file = os.path.join(self.data_dir, "sp100_quarterly_returns.csv")

        # Data containers
        self.tickers = []
        self.prices_df = None
        self.daily_returns_df = None
        self.quarterly_returns_df = None
        self.index_returns = None

        # Status flags
        self.data_loaded = False
        self.returns_computed = False

        # Log file path construction for debugging
        print(f"Data directory path: {self.data_dir}")
        print(f"Tickers file path: {self.tickers_file}")
        file_exists = os.path.exists(self.tickers_file)
        print(f"Tickers file exists: {file_exists}")



    def check_required_files(self) -> bool:
        """Check if required data files exist."""
        if not os.path.exists(self.tickers_file):
            logger.warning(f"Tickers file not found at {self.tickers_file}")
            return False
        return True

    def load_tickers(self) -> List[str]:
        """Load ticker symbols from CSV file."""
        if not os.path.exists(self.tickers_file):
            logger.error(f"Tickers file not found at {self.tickers_file}")
            return []

        try:
            tickers_df = pd.read_csv(self.tickers_file)
            if "Ticker" not in tickers_df.columns:
                logger.error("CSV must have a column named 'Ticker'")
                return []

            self.tickers = tickers_df["Ticker"].dropna().unique().tolist()
            logger.info(f"Loaded {len(self.tickers)} tickers")
            return self.tickers
        except Exception as e:
            logger.error(f"Error loading tickers: {e}")
            return []

    def download_stock_data(self, start_date: str, end_date: str, callback: Callable = None) -> pd.DataFrame:
        """
        Download historical stock data for the loaded tickers.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            callback: Optional callback function to report progress

        Returns:
            DataFrame with historical price data
        """
        if not self.tickers:
            self.load_tickers()
            if not self.tickers:
                if callback:
                    callback("No tickers found. Please check the tickers file.")
                return pd.DataFrame()

        data = {}
        total_tickers = len(self.tickers)

        for i, ticker in enumerate(self.tickers):
            if callback:
                callback(f"Downloading data for {ticker}... ({i+1}/{total_tickers})")

            try:
                ticker_data = yf.download(ticker, start=start_date, end=end_date,
                                          auto_adjust=False, progress=False)

                if ticker_data.empty:
                    if callback:
                        callback(f"Warning: No data for {ticker}. Skipping.")
                    continue

                if "Adj Close" not in ticker_data.columns:
                    if callback:
                        callback(f"Warning: 'Adj Close' missing for {ticker}. Skipping.")
                    continue

                adj_close = ticker_data["Adj Close"].squeeze()
                if adj_close.ndim != 1:
                    if callback:
                        callback(f"Error: Data for {ticker} is not 1D. Skipping.")
                    continue

                data[ticker] = adj_close

            except Exception as e:
                if callback:
                    callback(f"Error downloading {ticker}: {e}")

        if not data:
            if callback:
                callback("No valid data downloaded.")
            return pd.DataFrame()

        self.prices_df = pd.concat(data, axis=1)
        self.prices_df.to_csv(self.prices_file)

        if callback:
            callback(f"Data download complete. Saved to {self.prices_file}")

        self.data_loaded = True
        return self.prices_df

    def compute_returns(self, callback: Callable = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute daily and quarterly returns from price data.

        Args:
            callback: Optional callback function to report progress

        Returns:
            Tuple of (daily_returns_df, quarterly_returns_df)
        """
        if not self.data_loaded:
            if not os.path.exists(self.prices_file):
                if callback:
                    callback(f"Error: Prices file not found at {self.prices_file}")
                return None, None

            if callback:
                callback("Loading historical prices...")

            self.prices_df = pd.read_csv(self.prices_file, index_col=0, parse_dates=True)
            self.data_loaded = True

        if callback:
            callback("Computing daily returns...")

        # Calculate daily returns
        self.daily_returns_df = self.prices_df.pct_change().dropna()
        self.daily_returns_df.to_csv(self.daily_returns_file)

        if callback:
            callback("Computing quarterly returns...")

        # Calculate quarterly returns (compound returns)
        self.quarterly_returns_df = self.daily_returns_df.resample('Q').apply(
            lambda x: (1 + x).prod() - 1)
        self.quarterly_returns_df.to_csv(self.quarterly_returns_file)

        # Calculate benchmark index returns (equal-weighted)
        self.index_returns = self.quarterly_returns_df.mean(axis=1)

        if callback:
            callback("Returns computation complete.")
            callback(f"Daily returns saved to: {self.daily_returns_file}")
            callback(f"Quarterly returns saved to: {self.quarterly_returns_file}")

        self.returns_computed = True
        return self.daily_returns_df, self.quarterly_returns_df

    def load_returns(self, callback: Callable = None) -> bool:
        """Load precomputed returns from files."""
        if self.returns_computed:
            return True

        if not os.path.exists(self.quarterly_returns_file):
            if callback:
                callback(f"Error: Quarterly returns file not found at {self.quarterly_returns_file}")
            return False

        try:
            if callback:
                callback("Loading quarterly returns...")

            self.quarterly_returns_df = pd.read_csv(self.quarterly_returns_file,
                                                   index_col=0, parse_dates=True)

            # Calculate benchmark index returns (equal-weighted)
            self.index_returns = self.quarterly_returns_df.mean(axis=1)

            if callback:
                callback(f"Loaded returns for {self.quarterly_returns_df.shape[1]} stocks over "
                        f"{self.quarterly_returns_df.shape[0]} quarters")

            self.returns_computed = True
            return True

        except Exception as e:
            if callback:
                callback(f"Error loading returns: {e}")
            return False

class AMPLOptimizer:
    """Handles portfolio optimization using AMPL and Gurobi."""

    def __init__(self, data_manager: DataManager):
        """
        Initialize the AMPL optimizer.

        Args:
            data_manager: DataManager instance with loaded returns data
        """
        self.data_manager = data_manager
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(self.model_dir, exist_ok=True)

        # Create AMPL model file if it doesn't exist
        self.model_file = os.path.join(self.model_dir, "index_fund.mod")
        if not os.path.exists(self.model_file):
            self._create_ampl_model()

    def _create_ampl_model(self):
        """Create the AMPL model file for index fund optimization."""
        model_content = """
        # Index Fund Construction Model
        
        # Parameters
        param n;                       # Number of stocks
        param q;                       # Number of stocks to select
        param T;                       # Number of time periods
        
        param returns{1..n, 1..T};     # Stock returns
        param sp100_returns{1..T};     # S&P 100 index returns
        
        # Decision variables
        var select{1..n} binary;       # 1 if stock i is selected, 0 otherwise
        var weight{1..n} >= 0;         # Weight of stock i in the portfolio
        
        # Objective: minimize tracking error
        minimize tracking_error: 
            sum{t in 1..T} (
                (sum{i in 1..n} returns[i,t] * weight[i]) - sp100_returns[t]
            )^2;
        
        # Constraints
        # 1. Select exactly q stocks
        subject to select_q_stocks:
            sum{i in 1..n} select[i] = q;
        
        # 2. Weights sum to 1
        subject to sum_to_one:
            sum{i in 1..n} weight[i] = 1;
        
        # 3. Weight of stock i can be positive only if it is selected
        subject to link_weight_select{i in 1..n}:
            weight[i] <= select[i];
        """

        with open(self.model_file, 'w') as f:
            f.write(model_content)

        logger.info(f"Created AMPL model file at {self.model_file}")

    def optimize(self, q: int, callback: Callable = None) -> Dict:
        """
        Run AMPL optimization to select q stocks and their weights.

        Args:
            q: Number of stocks to select
            callback: Optional callback function for progress updates

        Returns:
            Dictionary with optimization results
        """
        if not self.data_manager.returns_computed:
            success = self.data_manager.load_returns(callback)
            if not success:
                if callback:
                    callback("Error: Returns data not available.")
                return None

        returns_df = self.data_manager.quarterly_returns_df
        index_returns = self.data_manager.index_returns

        try:
            if callback:
                callback("Initializing AMPL...")

            # Initialize AMPL
            ampl = AMPL()
            ampl.option["solver"] = "gurobi"

            # Read the model
            if callback:
                callback(f"Reading model from {self.model_file}...")
            ampl.read(self.model_file)

            # Prepare data for AMPL
            n = returns_df.shape[1]  # Number of stocks
            t = returns_df.shape[0]  # Number of time periods

            ampl.param['n'] = n
            ampl.param['q'] = q
            ampl.param['T'] = t

            if callback:
                callback(f"Preparing data for optimization (q={q}, n={n}, periods={t})...")

            # Set returns data
            returns_param = ampl.param["returns"]
            for i in range(n):
                for j in range(t):
                    returns_param[i+1, j+1] = returns_df.iloc[j, i]

            # Set index returns data
            sp100_param = ampl.param["sp100_returns"]
            for j in range(t):
                sp100_param[j+1] = index_returns.iloc[j]

            if callback:
                callback("Solving optimization problem...")

            # Solve the optimization problem
            ampl.solve()

            # Get the solve result status
            solve_result = ampl.get_value("solve_result")
            if callback:
                callback(f"Solve result: {solve_result}")

            # Extract results
            selected_stocks = {}
            select_var = ampl.get_variable('select')
            weight_var = ampl.get_variable('weight')

            for i in range(1, n+1):
                if select_var[i].value() > 0.5:  # If stock is selected
                    stock_ticker = returns_df.columns[i-1]
                    selected_stocks[stock_ticker] = weight_var[i].value()

            # Get objective value (tracking error)
            tracking_error = ampl.obj['tracking_error'].value()
            tracking_error_per_period = tracking_error / t

            if callback:
                callback(f"AMPL Optimization complete. Tracking Error: {tracking_error:.4e}")
                callback(f"Selected {len(selected_stocks)} stocks")

            # Calculate portfolio returns for selected stocks
            tickers = list(selected_stocks.keys())
            weights = list(selected_stocks.values())

            # Calculate portfolio returns for each period
            portfolio_returns = np.dot(returns_df[tickers].values, weights)

            # Calculate additional metrics
            sharpe = calculate_sharpe_ratio(portfolio_returns)
            mse = np.mean((portfolio_returns - index_returns.values) ** 2)
            correlation = np.corrcoef(portfolio_returns, index_returns.values)[0, 1]

            result = {
                'method': 'AMPL',
                'q': q,
                'selected_stocks': selected_stocks,
                'tickers': tickers,
                'weights': weights,
                'tracking_error': tracking_error,
                'tracking_error_per_period': tracking_error_per_period,
                'sharpe_ratio': sharpe,
                'mse': mse,
                'correlation': correlation,
                'portfolio_returns': portfolio_returns,
                'benchmark_returns': index_returns.values
            }

            # Save results to CSV
            output_file = os.path.join(self.data_manager.results_dir, f"ampl_q{q}.csv")
            pd.DataFrame({
                'Ticker': tickers,
                'Weight': weights
            }).to_csv(output_file, index=False)

            if callback:
                callback(f"Results saved to {output_file}")

            return result

        except Exception as e:
            if callback:
                callback(f"AMPL Optimization error: {e}")
            logger.error(f"AMPL Optimization error: {e}")
            return None

class GAOptimizer:
    """Handles portfolio optimization using Genetic Algorithm."""

    def __init__(self, data_manager: DataManager):
        """
        Initialize the GA optimizer.

        Args:
            data_manager: DataManager instance with loaded returns data
        """
        self.data_manager = data_manager
        self.fitness_history = []
        self.generation_callback = None

    def _fitness_func(self, ga_instance, solution, solution_idx):
        """
        Fitness function for the genetic algorithm.

        Evaluates how well a portfolio tracks the index by calculating
        the inverse of the tracking error.
        """
        returns_df = self.data_manager.quarterly_returns_df
        index_returns = self.data_manager.index_returns.values
        q = self.q

        # Select the q stocks with the highest gene values
        selected_idx = np.argsort(solution)[-q:]
        selected_genes = solution[selected_idx]

        # Compute weights using softmax
        weights = softmax(selected_genes)

        # Calculate portfolio returns
        portfolio_returns = np.dot(returns_df.iloc[:, selected_idx].values, weights)

        # Calculate tracking error (sum of squared differences)
        te = np.sum((portfolio_returns - index_returns) ** 2)

        # Return fitness (higher is better, so we use inverse of tracking error)
        return 1.0 / (te + 1e-8)

    def _on_generation(self, ga_instance):
        """Callback function called after each generation."""
        best_fitness = ga_instance.best_solution()[1]
        self.fitness_history.append(best_fitness)

        if self.generation_callback:
            self.generation_callback(
                f"Generation {ga_instance.generations_completed}, Best Fitness: {best_fitness:.2f}"
            )

    def optimize(self, q: int, num_generations: int = 100,
                 population_size: int = 50, callback: Callable = None) -> Dict:
        """
        Run GA optimization to select q stocks and their weights.

        Args:
            q: Number of stocks to select
            num_generations: Number of generations for the GA
            population_size: Population size per generation
            callback: Optional callback function for progress updates

        Returns:
            Dictionary with optimization results
        """
        if not self.data_manager.returns_computed:
            success = self.data_manager.load_returns(callback)
            if not success:
                if callback:
                    callback("Error: Returns data not available.")
                return None

        returns_df = self.data_manager.quarterly_returns_df
        index_returns = self.data_manager.index_returns

        # Reset fitness history
        self.fitness_history = []
        self.generation_callback = callback
        self.q = q

        try:
            if callback:
                callback(f"Setting up GA with q={q}, generations={num_generations}, population={population_size}")

            # Number of genes = number of stocks
            num_genes = returns_df.shape[1]

            # Define gene space (allowed values for genes)
            gene_space = {'low': -10, 'high': 10}

            # Create GA instance
            ga_instance = pygad.GA(
                num_generations=num_generations,
                sol_per_pop=population_size,
                num_parents_mating=population_size // 2,
                num_genes=num_genes,
                fitness_func=self._fitness_func,
                mutation_percent_genes=10,
                mutation_type="random",
                crossover_type="single_point",
                gene_space=gene_space,
                on_generation=self._on_generation,
                random_seed=42
            )

            if callback:
                callback("Running GA optimization...")

            # Run the GA
            ga_instance.run()

            # Get the best solution
            solution, solution_fitness, solution_idx = ga_instance.best_solution()

            if callback:
                callback(f"GA optimization complete. Best fitness: {solution_fitness:.2f}")

            # Process the best solution
            selected_idx = np.argsort(solution)[-q:]
            tickers = returns_df.columns[selected_idx].tolist()
            selected_genes = solution[selected_idx]
            weights = softmax(selected_genes)

            # Calculate portfolio returns
            portfolio_returns = np.dot(returns_df.iloc[:, selected_idx].values, weights)

            # Calculate metrics
            tracking_error = np.sum((portfolio_returns - index_returns.values) ** 2)
            tracking_error_per_period = tracking_error / len(index_returns)
            sharpe = calculate_sharpe_ratio(portfolio_returns)
            mse = np.mean((portfolio_returns - index_returns.values) ** 2)
            correlation = np.corrcoef(portfolio_returns, index_returns.values)[0, 1]

            if callback:
                callback(f"Tracking Error: {tracking_error:.4e}")
                callback(f"Selected {len(tickers)} stocks")

            result = {
                'method': 'GA',
                'q': q,
                'selected_stocks': dict(zip(tickers, weights)),
                'tickers': tickers,
                'weights': weights,
                'tracking_error': tracking_error,
                'tracking_error_per_period': tracking_error_per_period,
                'sharpe_ratio': sharpe,
                'mse': mse,
                'correlation': correlation,
                'portfolio_returns': portfolio_returns,
                'benchmark_returns': index_returns.values,
                'fitness_history': self.fitness_history
            }

            # Save results to CSV
            output_file = os.path.join(self.data_manager.results_dir, f"ga_q{q}.csv")
            pd.DataFrame({
                'Ticker': tickers,
                'Weight': weights
            }).to_csv(output_file, index=False)

            if callback:
                callback(f"Results saved to {output_file}")

            return result

        except Exception as e:
            if callback:
                callback(f"GA Optimization error: {e}")
            logger.error(f"GA Optimization error: {e}")
            return None

class VisualizationManager:
    """Handles creation and display of visualizations for portfolio analysis."""

    def __init__(self, data_manager: DataManager):
        """
        Initialize the visualization manager.

        Args:
            data_manager: DataManager instance with loaded data
        """
        self.data_manager = data_manager

    def plot_portfolio_composition(self, tickers: List[str], weights: List[float],
                                  ax=None, title: str = "Portfolio Composition") -> Figure:
        """
        Create a bar chart showing portfolio composition.

        Args:
            tickers: List of stock tickers
            weights: List of corresponding weights
            ax: Optional matplotlib axis to plot on
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#2b2b2b')
        else:
            fig = ax.figure

        # Sort by weight for better visualization
        sorted_data = sorted(zip(tickers, weights), key=lambda x: x[1], reverse=True)
        sorted_tickers, sorted_weights = zip(*sorted_data)

        # Create colorful bar chart
        cmap = get_cmap('viridis')
        colors = cmap(np.linspace(0.1, 0.9, len(tickers)))

        bars = ax.barh(sorted_tickers, sorted_weights, height=0.7, color=colors, alpha=0.8)

        # Add weight labels
        for i, bar in enumerate(bars):
            weight_pct = sorted_weights[i] * 100
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{weight_pct:.1f}%', va='center', color='white')

        # Style improvements
        ax.set_xlabel('Weight (%)', color='white', fontsize=12)
        ax.set_title(title, color='white', fontsize=14)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#555555')
        ax.spines['left'].set_color('#555555')
        ax.set_xlim(0, max(sorted_weights) * 1.15)

        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_interactive_performance_comparison(self, results: List[Dict],
                                                ax=None, title="Interactive Performance Comparison") -> Figure:
        """
        Create an interactive performance comparison plot.
        Plots the cumulative returns for each selected portfolio result along with the S&P 100 benchmark.

        Args:
            results: List of optimization result dictionaries to compare.
            ax: Optional matplotlib axis to plot on.
            title: Title for the plot.

        Returns:
            Matplotlib Figure.
        """
        # Create figure and axis if not provided.
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 8), facecolor='#2b2b2b')
        else:
            fig = ax.figure

        # Assume the benchmark (S&P 100) returns are identical across the selected results.
        benchmark_returns = np.array(results[0]['benchmark_returns'])
        dates = np.arange(len(benchmark_returns))

        # Plot the S&P 100 benchmark.
        benchmark_cum = (1 + benchmark_returns).cumprod()
        ax.plot(dates, benchmark_cum, label='S&P 100 Benchmark', linewidth=2, color='#ff7f0e')

        # Cycle through a set of colors for each portfolio.
        import itertools
        color_cycle = itertools.cycle(['#2ca02c', '#1f77b4', '#9467bd', '#8c564b', '#d62728', '#8c564b'])

        # Plot each selected portfolio result.
        for result in results:
            portfolio_returns = np.array(result['portfolio_returns'])
            portfolio_cum = (1 + portfolio_returns).cumprod()
            label = f"{result['method']} (q={result['q']})"
            ax.plot(dates, portfolio_cum, label=label, linewidth=2, color=next(color_cycle))

        # Format the plot.
        ax.set_title(title, color='white', fontsize=14)
        ax.set_xlabel('Period', color='white', fontsize=12)
        ax.set_ylabel('Cumulative Return', color='white', fontsize=12)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper left', facecolor='#2b2b2b', edgecolor='#555555', framealpha=0.8)

        # Attach interactive cursors with mplcursors.
        import mplcursors
        cursor = mplcursors.cursor(ax.lines, hover=True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(
            f"Period: {int(sel.target[0])}\nValue: {sel.target[1]:.2f}"
        ))

        plt.tight_layout()
        return fig

    def plot_performance_comparison(self, portfolio_returns: np.ndarray,
                                   benchmark_returns: np.ndarray,
                                   dates=None, ax=None,
                                   title: str = "Portfolio vs. S&P 100") -> Figure:
        """
        Create a line chart comparing portfolio performance to the benchmark.

        Args:
            portfolio_returns: Array of portfolio returns
            benchmark_returns: Array of benchmark returns
            dates: Optional array of dates for x-axis
            ax: Optional matplotlib axis to plot on
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#2b2b2b')
        else:
            fig = ax.figure

        # Convert returns to cumulative returns (1 + r)
        if dates is None:
            dates = np.arange(len(portfolio_returns))

        portfolio_cum = (1 + portfolio_returns).cumprod()
        benchmark_cum = (1 + benchmark_returns).cumprod()

        # Plot the cumulative returns
        ax.plot(dates, portfolio_cum, label='Portfolio', linewidth=2, color='#2ca02c')
        ax.plot(dates, benchmark_cum, label='S&P 100', linewidth=2, color='#ff7f0e')

        # Calculate final performance values
        final_portfolio = portfolio_cum[-1] - 1
        final_benchmark = benchmark_cum[-1] - 1

        # Add annotations for final values
        ax.annotate(f'{final_portfolio:.1%}',
                   xy=(dates[-1], portfolio_cum[-1]),
                   xytext=(5, 5), textcoords='offset points',
                   color='#2ca02c', fontweight='bold')

        ax.annotate(f'{final_benchmark:.1%}',
                   xy=(dates[-1], benchmark_cum[-1]),
                   xytext=(5, 5), textcoords='offset points',
                   color='#ff7f0e', fontweight='bold')

        # Style improvements
        ax.set_title(title, color='white', fontsize=14)
        ax.set_xlabel('Period', color='white', fontsize=12)
        ax.set_ylabel('Cumulative Return', color='white', fontsize=12)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#555555')
        ax.spines['left'].set_color('#555555')

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.3)

        # Add legend
        ax.legend(loc='upper left', facecolor='#2b2b2b', edgecolor='#555555', framealpha=0.8)

        plt.tight_layout()
        return fig

    def plot_fitness_evolution(self, fitness_history: List[float],
                              ax=None, title: str = "GA Optimization Progress") -> Figure:
        """
        Create a line chart showing GA fitness evolution.

        Args:
            fitness_history: List of fitness values over generations
            ax: Optional matplotlib axis to plot on
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#2b2b2b')
        else:
            fig = ax.figure

        generations = np.arange(1, len(fitness_history) + 1)

        # Plot fitness evolution
        ax.plot(generations, fitness_history, marker='o', markersize=4,
               linestyle='-', linewidth=2, color='#1f77b4')

        # Draw horizontal line at final fitness
        ax.axhline(y=fitness_history[-1], color='#d62728', linestyle='--', alpha=0.6)

        # Add annotation for final fitness
        ax.annotate(f'Final: {fitness_history[-1]:.2f}',
                   xy=(generations[-1], fitness_history[-1]),
                   xytext=(-60, 10), textcoords='offset points',
                   color='#d62728', fontweight='bold')

        # Style improvements
        ax.set_title(title, color='white', fontsize=14)
        ax.set_xlabel('Generation', color='white', fontsize=12)
        ax.set_ylabel('Fitness (1/Tracking Error)', color='white', fontsize=12)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#555555')
        ax.spines['left'].set_color('#555555')

        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_risk_return_profile(self, results: List[Dict],
                                ax=None, title: str = "Risk-Return Profile") -> Figure:
        """
        Create a scatter plot showing risk-return profile of different portfolios.

        Args:
            results: List of optimization result dictionaries
            ax: Optional matplotlib axis to plot on
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#2b2b2b')
        else:
            fig = ax.figure

        # Extract risk and return data
        risk = [np.sqrt(result['tracking_error_per_period']) for result in results]
        returns = [np.mean(result['portfolio_returns']) for result in results]
        labels = [f"{result['method']} (q={result['q']})" for result in results]

        # Create scatter plot
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for i, (x, y, label) in enumerate(zip(risk, returns, labels)):
            ax.scatter(x, y, s=100, color=colors[i % len(colors)], alpha=0.8, label=label)

        # Add benchmark
        benchmark_return = np.mean(results[0]['benchmark_returns'])
        ax.scatter(0, benchmark_return, s=150, color='gold', marker='*',
                  label='S&P 100 Benchmark')

        # Style improvements
        ax.set_title(title, color='white', fontsize=14)
        ax.set_xlabel('Risk (Tracking Error)', color='white', fontsize=12)
        ax.set_ylabel('Average Return', color='white', fontsize=12)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#555555')
        ax.spines['left'].set_color('#555555')

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.3)

        # Add legend
        ax.legend(loc='best', facecolor='#2b2b2b', edgecolor='#555555', framealpha=0.8)

        plt.tight_layout()
        return fig

    def plot_method_comparison(self, results: List[Dict],
                              ax=None, title: str = "Optimization Methods Comparison") -> Figure:
        """
        Create a bar chart comparing metrics across optimization methods.

        Args:
            results: List of optimization result dictionaries
            ax: Optional matplotlib axis to plot on
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#2b2b2b')
        else:
            fig = ax.figure

        # Extract data
        methods = [f"{result['method']} (q={result['q']})" for result in results]
        tracking_errors = [result['tracking_error_per_period'] for result in results]
        correlations = [result['correlation'] for result in results]
        sharpe_ratios = [result['sharpe_ratio'] for result in results]

        # Set bar positions
        x = np.arange(len(methods))
        width = 0.25

        # Create grouped bar chart
        ax.bar(x - width, tracking_errors, width, label='Tracking Error', alpha=0.7, color='#ff7f0e')
        ax.bar(x, correlations, width, label='Correlation', alpha=0.7, color='#2ca02c')
        ax.bar(x + width, sharpe_ratios, width, label='Sharpe Ratio', alpha=0.7, color='#1f77b4')

        # Style improvements
        ax.set_title(title, color='white', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, color='white', rotation=45, ha='right')
        ax.tick_params(axis='y', colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#555555')
        ax.spines['left'].set_color('#555555')

        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')

        # Add legend
        ax.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='#555555', framealpha=0.8)

        plt.tight_layout()
        return fig

    def create_dashboard(self, result: Dict, parent_frame) -> None:
        """
        Create a comprehensive dashboard visualization for a single optimization result.

        Args:
            result: Optimization result dictionary
            parent_frame: Parent frame to embed the visualization
        """
        # Clear existing widgets
        for widget in parent_frame.winfo_children():
            widget.destroy()

        # Create figure with subplots
        fig = plt.figure(figsize=(12, 10), facecolor='#2b2b2b')
        gs = GridSpec(3, 2, figure=fig)

        # Portfolio composition plot
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_portfolio_composition(result['tickers'], result['weights'],
                                       ax=ax1, title="Portfolio Composition")

        # Performance comparison plot
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_performance_comparison(result['portfolio_returns'],
                                        result['benchmark_returns'],
                                        ax=ax2, title="Portfolio vs. S&P 100")

        # Metrics visualization
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_metrics(result, ax=ax3)

        # GA fitness evolution (if available)
        if 'fitness_history' in result and result['fitness_history']:
            ax4 = fig.add_subplot(gs[2, :])
            self.plot_fitness_evolution(result['fitness_history'],
                                       ax=ax4, title="Optimization Progress")

        # Embed the figure in the frame
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)

    def _plot_metrics(self, result: Dict, ax=None) -> None:
        """Helper function to plot metrics."""
        metrics = {
            'Tracking Error': result['tracking_error_per_period'],
            'Correlation': result['correlation'],
            'Sharpe Ratio': result['sharpe_ratio'],
            'Stocks Selected': len(result['tickers']),
            'MSE': result['mse']
        }

        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        # Create horizontal bar chart
        bars = ax.barh(metric_names, metric_values, height=0.5, color='#1f77b4')

        # Add value labels
        for i, bar in enumerate(bars):
            value = metric_values[i]
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', va='center', color='white')

        # Style improvements
        ax.set_title("Portfolio Metrics", color='white', fontsize=14)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#555555')
        ax.spines['left'].set_color('#555555')

        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.3, axis='x')

class EnhancedGUI(ctk.CTk):
    """Main application GUI"""

    def __init__(self):
        """Initialize the GUI application."""
        super().__init__()

        # Set window properties
        self.title("Enhanced Index Fund Constructor")
        self.geometry("1400x900")

        self.interactive_var = BooleanVar(master=self, value=False)

        # Initialize components
        self.data_manager = DataManager()
        self.ampl_optimizer = AMPLOptimizer(self.data_manager)
        self.ga_optimizer = GAOptimizer(self.data_manager)
        self.viz_manager = VisualizationManager(self.data_manager)

        # Store optimization results
        self.optimization_results = []

        # Create GUI layout
        self.setup_ui()



    def setup_ui(self):
        """Create the main application UI."""
        # Configure grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create main tabview
        self.tab_view = ctk.CTkTabview(self, corner_radius=10)
        self.tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Create tabs
        self.data_tab = self.tab_view.add("Data Management")
        self.optim_tab = self.tab_view.add("Optimization")
        self.viz_tab = self.tab_view.add("Visualization")
        self.comp_tab = self.tab_view.add("Comparison")
        self.settings_tab = self.tab_view.add("Settings")

        # Set up each tab
        self.setup_data_tab()
        self.setup_optimization_tab()
        self.setup_visualization_tab()
        self.setup_comparison_tab()
        self.setup_settings_tab()

    def setup_data_tab(self):
        """Set up the Data Management tab."""
        # Create frames
        control_frame = ctk.CTkFrame(self.data_tab, corner_radius=10)
        control_frame.pack(padx=10, pady=10, fill="x")

        data_frame = ctk.CTkFrame(self.data_tab, corner_radius=10)
        data_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Create data control widgets
        date_frame = ctk.CTkFrame(control_frame, corner_radius=10)
        date_frame.pack(side="left", padx=10, pady=10)

        ctk.CTkLabel(date_frame, text="Start Date:").grid(row=0, column=0, padx=5, pady=5)
        self.start_date_entry = ctk.CTkEntry(date_frame, placeholder_text="YYYY-MM-DD")
        self.start_date_entry.grid(row=0, column=1, padx=5, pady=5)
        self.start_date_entry.insert(0, "2023-01-01")

        ctk.CTkLabel(date_frame, text="End Date:").grid(row=1, column=0, padx=5, pady=5)
        self.end_date_entry = ctk.CTkEntry(date_frame, placeholder_text="YYYY-MM-DD")
        self.end_date_entry.grid(row=1, column=1, padx=5, pady=5)
        self.end_date_entry.insert(0, "2025-03-01")

        # Create action buttons
        btn_frame = ctk.CTkFrame(control_frame, corner_radius=10)
        btn_frame.pack(side="right", padx=10, pady=10)

        ctk.CTkButton(
            btn_frame, text="Download Data", command=self.download_data,
            fg_color="#2a9d8f", hover_color="#264653"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame, text="Compute Returns", command=self.compute_returns,
            fg_color="#e76f51", hover_color="#e63946"
        ).pack(side="left", padx=5)

        # Create log area with a header
        log_frame = ctk.CTkFrame(data_frame, corner_radius=10)
        log_frame.pack(side="left", padx=10, pady=10, fill="both", expand=True)

        ctk.CTkLabel(log_frame, text="Process Log", font=("Arial", 14, "bold")).pack(pady=5)

        self.data_log = ctk.CTkTextbox(log_frame, height=400, width=400, font=("Consolas", 12))
        self.data_log.pack(padx=10, pady=10, fill="both", expand=True)

        # Create data preview area
        preview_frame = ctk.CTkFrame(data_frame, corner_radius=10)
        preview_frame.pack(side="right", padx=10, pady=10, fill="both", expand=True)

        ctk.CTkLabel(preview_frame, text="Data Preview", font=("Arial", 14, "bold")).pack(pady=5)

        self.data_preview = ctk.CTkTextbox(preview_frame, height=400, width=400, font=("Consolas", 12))
        self.data_preview.pack(padx=10, pady=10, fill="both", expand=True)

    def setup_optimization_tab(self):
        """Set up the Optimization tab."""
        # Create top frame for controls
        control_frame = ctk.CTkFrame(self.optim_tab, corner_radius=10)
        control_frame.pack(padx=10, pady=10, fill="x")

        # Create parameters frame
        param_frame = ctk.CTkFrame(control_frame, corner_radius=10)
        param_frame.pack(side="left", padx=10, pady=10, fill="x", expand=True)

        # Stock selection parameters
        stock_frame = ctk.CTkFrame(param_frame, corner_radius=10)
        stock_frame.pack(padx=10, pady=10, fill="x")

        ctk.CTkLabel(stock_frame, text="Number of Stocks (q):").pack(side="left", padx=5)
        self.q_entry = ctk.CTkEntry(stock_frame, width=80)
        self.q_entry.pack(side="left", padx=5)
        self.q_entry.insert(0, "20")

        # GA parameters
        ga_frame = ctk.CTkFrame(param_frame, corner_radius=10)
        ga_frame.pack(padx=10, pady=10, fill="x")

        ctk.CTkLabel(ga_frame, text="GA Generations:").pack(side="left", padx=5)
        self.gen_entry = ctk.CTkEntry(ga_frame, width=80)
        self.gen_entry.pack(side="left", padx=5)
        self.gen_entry.insert(0, "100")

        ctk.CTkLabel(ga_frame, text="Population Size:").pack(side="left", padx=5)
        self.pop_entry = ctk.CTkEntry(ga_frame, width=80)
        self.pop_entry.pack(side="left", padx=5)
        self.pop_entry.insert(0, "50")

        # Run buttons
        btn_frame = ctk.CTkFrame(control_frame, corner_radius=10)
        btn_frame.pack(side="right", padx=10, pady=10)

        ctk.CTkButton(
            btn_frame, text="Run AMPL Optimization", command=self.run_ampl_optimization,
            fg_color="#2a9d8f", hover_color="#264653", width=200
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame, text="Run GA Optimization", command=self.run_ga_optimization,
            fg_color="#e76f51", hover_color="#e63946", width=200
        ).pack(side="left", padx=5)

        # Create split view for log and results
        content_frame = ctk.CTkFrame(self.optim_tab, corner_radius=10)
        content_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Log area
        log_frame = ctk.CTkFrame(content_frame, corner_radius=10)
        log_frame.pack(side="left", padx=10, pady=10, fill="both", expand=True)

        ctk.CTkLabel(log_frame, text="Optimization Log", font=("Arial", 14, "bold")).pack(pady=5)

        self.optim_log = ctk.CTkTextbox(log_frame, height=400, width=300, font=("Consolas", 12))
        self.optim_log.pack(padx=10, pady=10, fill="both", expand=True)

        # Results visualization area
        viz_frame = ctk.CTkFrame(content_frame, corner_radius=10)
        viz_frame.pack(side="right", padx=10, pady=10, fill="both", expand=True)

        ctk.CTkLabel(viz_frame, text="Optimization Results", font=("Arial", 14, "bold")).pack(pady=5)

        self.optim_viz_frame = ctk.CTkFrame(viz_frame, corner_radius=10)
        self.optim_viz_frame.pack(padx=10, pady=10, fill="both", expand=True)

    def setup_visualization_tab(self):
        # Create control frame
        control_frame = ctk.CTkFrame(self.viz_tab, corner_radius=10)
        control_frame.pack(padx=10, pady=10, fill="x")

        # Dropdown to select result to visualize
        ctk.CTkLabel(control_frame, text="Select Optimization Result:").pack(side="left", padx=5)

        self.result_var = StringVar()
        self.result_dropdown = ctk.CTkOptionMenu(
            control_frame, variable=self.result_var,
            values=["No results available"],
            command=self.update_visualization
        )
        self.result_dropdown.pack(side="left", padx=5)

        # Export button
        ctk.CTkButton(
            control_frame, text="Export Visualization", command=self.export_visualization,
            fg_color="#2a9d8f", hover_color="#264653"
        ).pack(side="right", padx=5)



        # Visualization canvas
        self.viz_canvas_frame = ctk.CTkFrame(self.viz_tab, corner_radius=10)
        self.viz_canvas_frame.pack(padx=10, pady=10, fill="both", expand=True)

    def setup_comparison_tab(self):
        """Set up the Comparison tab (without radio buttons)."""
        # Create the top control frame for comparison options
        control_frame = ctk.CTkFrame(self.comp_tab, corner_radius=10)
        control_frame.pack(padx=10, pady=10, fill="x")

        # Left frame for the multi-select list
        left_frame = ctk.CTkFrame(control_frame, corner_radius=10)
        left_frame.pack(side="left", padx=10, pady=10, fill="both", expand=True)

        # Label instructing the user to pick multiple results
        ctk.CTkLabel(
            left_frame,
            text="Select one or more Results to Compare:",
            font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=5, pady=5)

        # Use a standard Tkinter Listbox in a CTkFrame for multi-selection
        import tkinter as tk
        self.results_listbox = tk.Listbox(
            left_frame,
            selectmode="extended",  # Allows multi-select
            bg="#2b2b2b",  # Dark background to match CTk theme
            fg="white",  # Text color
            highlightbackground="#555555",
            font=("Arial", 12)
        )
        self.results_listbox.pack(padx=5, pady=5, fill="both", expand=True)

        # Right frame for the comparison metric dropdown and compare button
        right_frame = ctk.CTkFrame(control_frame, corner_radius=10)
        right_frame.pack(side="left", padx=10, pady=10, fill="y")

        # Label for the comparison metric
        ctk.CTkLabel(
            right_frame,
            text="Comparison Metric:",
            font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=5, pady=5)

        # Dropdown to choose the metric
        self.comparison_metric_var = StringVar(value="All Comparisons")
        options = [
            "All Comparisons",
            "Method Comparison",
            "Risk-Return Profile",
            "Performance Comparison",
            "Stock Frequency"
        ]
        ctk.CTkOptionMenu(
            right_frame,
            variable=self.comparison_metric_var,
            values=options
        ).pack(pady=5)

        # Button to trigger the comparison
        ctk.CTkButton(
            right_frame,
            text="Generate Comparison",
            command=self.generate_comparison,
            fg_color="#2a9d8f",
            hover_color="#264653"
        ).pack(pady=5)

        # Frame where the comparison figure(s) will be displayed
        self.comp_canvas_frame = ctk.CTkFrame(self.comp_tab, corner_radius=10)
        self.comp_canvas_frame.pack(padx=10, pady=10, fill="both", expand=True)

    def setup_settings_tab(self):
        """Set up the Settings tab."""
        # Create settings frame
        settings_frame = ctk.CTkFrame(self.settings_tab, corner_radius=10)
        settings_frame.pack(padx=20, pady=20, fill="both", expand=True)

        # Theme settings
        theme_frame = ctk.CTkFrame(settings_frame, corner_radius=10)
        theme_frame.pack(padx=10, pady=10, fill="x")

        ctk.CTkLabel(theme_frame, text="Application Theme:", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)

        theme_options = ["Dark", "Light", "System"]
        self.theme_var = StringVar(value="Dark")

        for theme in theme_options:
            ctk.CTkRadioButton(
                theme_frame, text=theme, variable=self.theme_var, value=theme,
                command=self.change_theme
            ).pack(anchor="w", padx=20, pady=5)

        # Data directory settings
        dir_frame = ctk.CTkFrame(settings_frame, corner_radius=10)
        dir_frame.pack(padx=10, pady=10, fill="x")

        ctk.CTkLabel(dir_frame, text="Data Directory:", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)

        dir_select_frame = ctk.CTkFrame(dir_frame, corner_radius=0)
        dir_select_frame.pack(padx=20, pady=5, fill="x")

        self.dir_entry = ctk.CTkEntry(dir_select_frame, width=300)
        self.dir_entry.pack(side="left", padx=5, fill="x", expand=True)
        self.dir_entry.insert(0, self.data_manager.data_dir)

        ctk.CTkButton(
            dir_select_frame, text="Browse", command=self.browse_data_dir,
            width=100
        ).pack(side="right", padx=5)

        # About section
        about_frame = ctk.CTkFrame(settings_frame, corner_radius=10)
        about_frame.pack(padx=10, pady=10, fill="x")

        ctk.CTkLabel(about_frame, text="About", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)

        about_text = """
        Enhanced Index Fund Constructor (EIFC)
        
        A comprehensive application for creating optimal index fund portfolios 
        that track the S&P 100 using fewer stocks.
        
        Created: April 12, 2025
        Lee Pettigrew
        X20730039
        """

        ctk.CTkLabel(about_frame, text=about_text, justify="left").pack(anchor="w", padx=20, pady=5)

    # Data tab functions
    def download_data(self):
        """Download stock data."""
        self.clear_log(self.data_log)
        self.log_data("Starting data download process...")

        start_date = self.start_date_entry.get()
        end_date = self.end_date_entry.get()

        if not self.validate_dates(start_date, end_date):
            self.log_data("Error: Invalid date format. Use YYYY-MM-DD.")
            return

        self.log_data(f"Date range: {start_date} to {end_date}")

        # Run in a separate thread to keep UI responsive
        threading.Thread(
            target=self._download_data_thread,
            args=(start_date, end_date),
            daemon=True
        ).start()

    def _download_data_thread(self, start_date, end_date):
        """Background thread for data download."""
        if not self.data_manager.check_required_files():
            self.log_data("Error: Missing tickers file. Please create sp100_tickers.csv in the data directory.")
            return

        self.log_data("Loading tickers...")
        tickers = self.data_manager.load_tickers()

        if not tickers:
            self.log_data("Error: No tickers found.")
            return

        self.log_data(f"Found {len(tickers)} tickers. Starting download...")

        # Download data
        prices_df = self.data_manager.download_stock_data(start_date, end_date, self.log_data)

        if prices_df.empty:
            self.log_data("Error: Failed to download stock data.")
            return

        # Display data preview
        self.update_data_preview(prices_df)

        self.log_data("Data download complete!")

    def compute_returns(self):
        """Compute daily and quarterly returns."""
        self.clear_log(self.data_log)
        self.log_data("Starting returns computation...")

        # Run in a separate thread to keep UI responsive
        threading.Thread(
            target=self._compute_returns_thread,
            daemon=True
        ).start()

    def _compute_returns_thread(self):
        """Background thread for returns computation."""
        daily_returns, quarterly_returns = self.data_manager.compute_returns(self.log_data)

        if daily_returns is None or quarterly_returns is None:
            self.log_data("Error: Failed to compute returns.")
            return

        # Display data preview
        self.update_data_preview(quarterly_returns)

        self.log_data("Returns computation complete!")

    # Optimization tab functions
    def run_ampl_optimization(self):
        """Run AMPL optimization."""
        self.clear_log(self.optim_log)
        self.log_optim("Starting AMPL optimization...")

        try:
            q = int(self.q_entry.get())
            if q <= 0:
                raise ValueError("q must be positive")
        except ValueError as e:
            self.log_optim(f"Error: Invalid q value. {e}")
            return

        self.log_optim(f"Running optimization with q = {q}")

        # Run in a separate thread to keep UI responsive
        threading.Thread(
            target=self._run_ampl_thread,
            args=(q,),
            daemon=True
        ).start()

    def _run_ampl_thread(self, q):
        """Background thread for AMPL optimization."""
        # Check if returns data is available
        if not self.data_manager.returns_computed:
            success = self.data_manager.load_returns(self.log_optim)
            if not success:
                self.log_optim("Error: No returns data available. Compute returns first.")
                return

        # Run optimization
        result = self.ampl_optimizer.optimize(q, self.log_optim)

        if result is None:
            self.log_optim("Error: AMPL optimization failed.")
            return

        # Store result
        self.optimization_results.append(result)

        # Update result dropdown in visualization tab
        self.update_result_dropdown()

        # Display visualization
        self.viz_manager.create_dashboard(result, self.optim_viz_frame)

        self.log_optim("AMPL optimization complete!")

    def run_ga_optimization(self):
        """Run Genetic Algorithm optimization."""
        self.clear_log(self.optim_log)
        self.log_optim("Starting GA optimization...")

        try:
            q = int(self.q_entry.get())
            if q <= 0:
                raise ValueError("q must be positive")

            generations = int(self.gen_entry.get())
            if generations <= 0:
                raise ValueError("generations must be positive")

            population = int(self.pop_entry.get())
            if population <= 0:
                raise ValueError("population must be positive")

        except ValueError as e:
            self.log_optim(f"Error: Invalid parameter value. {e}")
            return

        self.log_optim(f"Running GA optimization with q = {q}, generations = {generations}, population = {population}")

        # Run in a separate thread to keep UI responsive
        threading.Thread(
            target=self._run_ga_thread,
            args=(q, generations, population),
            daemon=True
        ).start()

    def _run_ga_thread(self, q, generations, population):
        """Background thread for GA optimization."""
        # Check if returns data is available
        if not self.data_manager.returns_computed:
            success = self.data_manager.load_returns(self.log_optim)
            if not success:
                self.log_optim("Error: No returns data available. Compute returns first.")
                return

        # Run optimization
        result = self.ga_optimizer.optimize(q, generations, population, self.log_optim)

        if result is None:
            self.log_optim("Error: GA optimization failed.")
            return

        # Store result
        self.optimization_results.append(result)

        # Update result dropdown in visualization tab
        self.update_result_dropdown()

        # Display visualization
        self.viz_manager.create_dashboard(result, self.optim_viz_frame)

        self.log_optim("GA optimization complete!")

    # Visualization tab functions
    def update_visualization(self, selected=None):
        if not self.optimization_results:
            return
        selected_value = self.result_var.get()
        if selected_value == "No results available":
            return
        import re
        match = re.search(r"Result (\d+):", selected_value)
        if not match:
            return
        idx = int(match.group(1)) - 1
        result = self.optimization_results[idx]
        self.viz_manager.create_dashboard(result, self.viz_canvas_frame)

    def export_visualization(self):
        if not self.optimization_results:
            messagebox.showinfo("Export", "No results available to export.")
            return

        selected_value = self.result_var.get()
        import re
        match = re.search(r"Result (\d+):", selected_value)
        if not match:
            messagebox.showerror("Export", "Error getting selected result.")
            return
        idx = int(match.group(1)) - 1
        result = self.optimization_results[idx]

        fig = plt.figure(figsize=(14, 12), facecolor='#2b2b2b')
        gs = GridSpec(3, 2, figure=fig)

        # Portfolio Composition Plot
        ax1 = fig.add_subplot(gs[0, 0])
        self.viz_manager.plot_portfolio_composition(result['tickers'], result['weights'],
                                                    ax=ax1, title="Portfolio Composition")

        # Performance Comparison Plot
        ax2 = fig.add_subplot(gs[0, 1])
        if self.interactive_var.get():
            self.viz_manager.plot_interactive_performance_comparison(result['portfolio_returns'],
                                                                     result['benchmark_returns'], ax=ax2)
        else:
            self.viz_manager.plot_performance_comparison(result['portfolio_returns'],
                                                         result['benchmark_returns'], ax=ax2,
                                                         title="Portfolio vs. S&P 100")

        # Metrics Visualization
        ax3 = fig.add_subplot(gs[1, :])
        self.viz_manager._plot_metrics(result, ax=ax3)

        # GA Fitness Evolution (if available)
        if 'fitness_history' in result and result['fitness_history']:
            ax4 = fig.add_subplot(gs[2, :])
            self.viz_manager.plot_fitness_evolution(result['fitness_history'],
                                                    ax=ax4, title="Optimization Progress")

        plt.tight_layout()
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"), ("SVG Image", "*.svg")]
        )
        if not file_path:
            return
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        messagebox.showinfo("Export", f"Visualization exported to {file_path}")



    # Comparison tab functions
    def generate_comparison(self):
        """Generate the comparison visualization based on selected metric."""
        if len(self.optimization_results) < 2:
            messagebox.showinfo("Comparison", "Need at least 2 optimization results to compare.")
            return

        # Get selected indices from the listbox (if none are selected, default to all results)
        selected_indices = self.results_listbox.curselection()
        if selected_indices:
            selected_results = [self.optimization_results[int(i)] for i in selected_indices]
        else:
            selected_results = self.optimization_results

        # Clear any existing widgets in the comparison canvas
        for widget in self.comp_canvas_frame.winfo_children():
            widget.destroy()

        metric = self.comparison_metric_var.get()

        if metric == "All Comparisons":
            fig = plt.figure(figsize=(14, 12), facecolor='#2b2b2b')
            gs = GridSpec(2, 2, figure=fig)
            # Top-left: Method Comparison
            ax1 = fig.add_subplot(gs[0, 0])
            self.viz_manager.plot_method_comparison(selected_results, ax=ax1, title="Optimization Methods Comparison")
            # Top-right: Risk-Return Profile
            ax2 = fig.add_subplot(gs[0, 1])
            self.viz_manager.plot_risk_return_profile(selected_results, ax=ax2, title="Risk-Return Profile")
            # Bottom-left: Interactive Performance Comparison
            ax3 = fig.add_subplot(gs[1, 0])
            self.viz_manager.plot_interactive_performance_comparison(
                selected_results, ax=ax3, title="Interactive Performance Comparison")

            # Bottom-right: Stock Selection Frequency
            ax4 = fig.add_subplot(gs[1, 1])
            stock_counts = {}
            for result in selected_results:
                for ticker in result['tickers']:
                    stock_counts[ticker] = stock_counts.get(ticker, 0) + 1
            sorted_stocks = sorted(stock_counts.items(), key=lambda x: x[1], reverse=True)
            top_stocks = sorted_stocks[:15]
            top_tickers = [s[0] for s in top_stocks]
            top_counts = [s[1] for s in top_stocks]
            ax4.barh(top_tickers, top_counts, height=0.7, color='#1f77b4', alpha=0.8)
            ax4.set_title("Most Selected Stocks", color='white', fontsize=14)
            ax4.set_xlabel('Frequency', color='white', fontsize=12)
            ax4.tick_params(axis='x', colors='white')
            ax4.tick_params(axis='y', colors='white')
            ax4.grid(True, linestyle='--', alpha=0.3, axis='x')
            plt.tight_layout()
        elif metric == "Method Comparison":
            fig = plt.figure(figsize=(12, 8), facecolor='#2b2b2b')
            self.viz_manager.plot_method_comparison(selected_results, ax=plt.gca(),
                                                    title="Optimization Methods Comparison")
        elif metric == "Risk-Return Profile":
            fig = plt.figure(figsize=(12, 8), facecolor='#2b2b2b')
            self.viz_manager.plot_risk_return_profile(selected_results, ax=plt.gca(), title="Risk-Return Profile")

        elif metric == "Performance Comparison":
            fig = plt.figure(figsize=(12, 8), facecolor='#2b2b2b')
            self.viz_manager.plot_interactive_performance_comparison(
                selected_results, ax=plt.gca(), title="Interactive Performance Comparison")


        elif metric == "Stock Frequency":
            fig = plt.figure(figsize=(12, 8), facecolor='#2b2b2b')
            stock_counts = {}
            for result in selected_results:
                for ticker in result['tickers']:
                    stock_counts[ticker] = stock_counts.get(ticker, 0) + 1
            sorted_stocks = sorted(stock_counts.items(), key=lambda x: x[1], reverse=True)
            top_stocks = sorted_stocks[:15]
            top_tickers = [s[0] for s in top_stocks]
            top_counts = [s[1] for s in top_stocks]
            plt.barh(top_tickers, top_counts, height=0.7, color='#1f77b4', alpha=0.8)
            plt.title("Most Selected Stocks", color='white', fontsize=14)
            plt.xlabel("Frequency", color='white', fontsize=12)
            plt.xticks(color='white')
            plt.yticks(color='white')
            plt.grid(True, linestyle='--', alpha=0.3, axis='x')
        else:
            fig = plt.figure(figsize=(12, 8), facecolor='#2b2b2b')
            plt.text(0.5, 0.5, "Invalid Comparison Metric", color='white', ha='center', va='center')

        # Embed the created figure into the comparison canvas in the GUI
        canvas = FigureCanvasTkAgg(fig, master=self.comp_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # Settings tab functions
    def change_theme(self):
        """Change application theme."""
        theme = self.theme_var.get()
        ctk.set_appearance_mode(theme)

    def browse_data_dir(self):
        """Browse for data directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.dir_entry.delete(0, 'end')
            self.dir_entry.insert(0, directory)

            # Update data manager paths
            self.data_manager.data_dir = directory
            self.data_manager.tickers_file = os.path.join(directory, "sp100_tickers.csv")
            self.data_manager.prices_file = os.path.join(directory, "sp100_prices.csv")
            self.data_manager.daily_returns_file = os.path.join(directory, "sp100_daily_returns.csv")
            self.data_manager.quarterly_returns_file = os.path.join(directory, "sp100_quarterly_returns.csv")

    # Helper functions
    def log_data(self, message):
        """Log message to the data tab."""
        self.data_log.insert("end", f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.data_log.see("end")
        self.update_idletasks()

    def log_optim(self, message):
        """Log message to the optimization tab."""
        self.optim_log.insert("end", f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.optim_log.see("end")
        self.update_idletasks()

    def clear_log(self, log_widget):
        """Clear log widget."""
        log_widget.delete("1.0", "end")

    def update_data_preview(self, dataframe):
        """Update data preview with DataFrame."""
        self.data_preview.delete("1.0", "end")

        # Show shape and info
        self.data_preview.insert("end", f"Shape: {dataframe.shape[0]} rows × {dataframe.shape[1]} columns\n\n")

        # Show column names
        self.data_preview.insert("end", "Columns: " + ", ".join(dataframe.columns[:10]))
        if len(dataframe.columns) > 10:
            self.data_preview.insert("end", "... and more\n\n")
        else:
            self.data_preview.insert("end", "\n\n")

        # Show preview of the data
        self.data_preview.insert("end", "Data Preview:\n")
        self.data_preview.insert("end", dataframe.head().to_string())

    def update_result_dropdown(self):
        """Update the results dropdown and the multi-select listbox in the visualization tab."""
        import tkinter as tk
        if not self.optimization_results:
            return

        # Create descriptive text for each result.
        values = [f"Result {i + 1}: {result['method']} (q={result['q']})"
                  for i, result in enumerate(self.optimization_results)]

        # Update the OptionMenu dropdown.
        self.result_dropdown.configure(values=values)
        self.result_var.set(values[-1])  # Select the most recent result

        # Update the multi-select Listbox with the descriptive names.
        self.results_listbox.delete(0, tk.END)
        for entry in values:
            self.results_listbox.insert(tk.END, entry)

    def validate_dates(self, start_date, end_date):
        """Validate date strings are in correct format."""
        try:
            datetime.datetime.strptime(start_date, "%Y-%m-%d")
            datetime.datetime.strptime(end_date, "%Y-%m-%d")
            return True
        except ValueError:
            return False

def main():
    """Main function to run the application."""
    app = EnhancedGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
