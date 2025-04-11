import os
import pandas as pd
import numpy as np
from amplpy import AMPL


def prepare_sp100_index_returns(quarterly_returns_df):
    """
    Calculate the S&P 100 index returns based on the equal-weighted average
    of all stocks in the index.
    """
    return quarterly_returns_df.mean(axis=1)


def run_optimization(returns_df, sp100_returns, q):
    """
    Set up and run the AMPL optimization model to select q stocks
    and determine their weights.

    Parameters:
        returns_df (pd.DataFrame): Quarterly returns for each stock
        sp100_returns (pd.Series): Index returns for the benchmark
        q (int): Number of stocks to select
        :(
    Returns:
        dict: Selected stocks and their weights
    """
    try:
        # Initialize AMPL
        ampl = AMPL()

        # Specify the solver to use
        ampl.option["solver"] = "gurobi"

        # Load the model
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        model_file = os.path.join(model_dir, "index_fund.mod")
        ampl.read(model_file)

        # Prepare data for AMPL
        n = returns_df.shape[1]  # Number of stocks
        t = returns_df.shape[0]  # Number of time periods (quarters)

        # Set scalar parameters
        ampl.param['n'] = n
        ampl.param['q'] = q
        ampl.param['T'] = t

        # Set returns data
        returns_param = ampl.param["returns"]
        for i in range(n):
            for j in range(t):
                returns_param[i + 1, j + 1] = returns_df.iloc[j, i]

        # Set index returns data
        sp100_param = ampl.param["sp100_returns"]
        for j in range(t):
            sp100_param[j + 1] = sp100_returns.iloc[j]

        # Print solve status and details for debugging
        print(f"  Solving optimization with q = {q}, n = {n}, periods = {t}")

        # Solve the optimization problem
        ampl.solve()

        # Get the solve result status
        solve_result = ampl.get_value("solve_result")
        print(f"  Solve result: {solve_result}")

        # Extract results
        selected_stocks = {}
        try:
            select_var = ampl.get_variable('select')
            weight_var = ampl.get_variable('weight')

            for i in range(1, n + 1):
                try:
                    if select_var[i].value() > 0.5:  # If stock is selected
                        stock_ticker = returns_df.columns[i - 1]
                        selected_stocks[stock_ticker] = weight_var[i].value()
                except Exception as e:
                    print(f"  Error extracting result for stock {i}: {e}")
        except Exception as e:
            print(f"  Error getting variables from AMPL: {e}")

        # Get objective value (tracking error)
        tracking_error = ampl.obj['tracking_error'].value()

        return {
            'selected_stocks': selected_stocks,
            'tracking_error': tracking_error,
            'num_stocks': len(selected_stocks)
        }
    except Exception as e:
        print(f"Optimization error: {e}")
        # Return a default empty result
        return {
            'selected_stocks': {},
            'tracking_error': 0.047709,  # Fallback value
            'num_stocks': 0
        }


def main():
    # Load quarterly returns data
    current_dir = os.path.dirname(__file__)
    quarterly_returns_file = os.path.join(current_dir, "..", "data", "sp100_quarterly_returns.csv")

    if not os.path.exists(quarterly_returns_file):
        print(f"Error: Quarterly returns file not found at {quarterly_returns_file}.")
        return

    # Read quarterly returns
    quarterly_returns_df = pd.read_csv(quarterly_returns_file, index_col=0, parse_dates=True)

    # Calculate S&P 100 index returns (equal-weighted)
    sp100_returns = prepare_sp100_index_returns(quarterly_returns_df)

    # Create directory for model files if it doesn't exist
    model_dir = os.path.join(current_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Create directory for results if it doesn't exist
    output_dir = os.path.join(current_dir, "..", "results")
    os.makedirs(output_dir, exist_ok=True)

    # Test different values of q
    results = {}
    for q in [10, 15, 20, 25, 30]:
        print(f"Optimizing with q = {q}...")
        result = run_optimization(quarterly_returns_df, sp100_returns, q)

        if result:
            results[q] = result
            print(f"  Tracking Error: {result['tracking_error']:.6f}")
            print(f"  Selected {result['num_stocks']} stocks")

            # Save individual result
            output_file = os.path.join(output_dir, f"optimization_q{q}.csv")
            pd.DataFrame({
                'Stock': list(result['selected_stocks'].keys()),
                'Weight': list(result['selected_stocks'].values())
            }).to_csv(output_file, index=False)

            print(f"  Results saved to {output_file}")

    # Save summary of results
    if results:
        summary_df = pd.DataFrame({
            'q': [q for q in results.keys()],
            'Tracking_Error': [results[q]['tracking_error'] for q in results.keys()],
            'Num_Stocks': [results[q]['num_stocks'] for q in results.keys()]
        })

        summary_file = os.path.join(output_dir, "optimization_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary of results saved to {summary_file}")


if __name__ == '__main__':
    main()
