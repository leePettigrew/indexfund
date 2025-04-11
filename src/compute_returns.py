import os
import pandas as pd


def compute_daily_returns(prices_df):
    """
    Compute daily percentage returns from prices.

    Parameters:
        prices_df (pd.DataFrame): DataFrame containing historical prices with DateTime index.

    Returns:
        pd.DataFrame: Daily returns.
    """
    return prices_df.pct_change().dropna()


def compute_periodic_returns(daily_returns, period='Q'):
    """
    Compute periodic returns by resampling the daily returns.

    Parameters:
        daily_returns (pd.DataFrame): DataFrame containing daily returns.
        period (str): Resampling period ('Q' for quarterly, 'M' for monthly, etc.)

    Returns:
        pd.DataFrame: Periodic returns computed as the compounded return.
    """
    # Compounded return: (1 + r1) * (1 + r2) * ... - 1
    periodic_returns = daily_returns.resample(period).apply(lambda x: (1 + x).prod() - 1)
    return periodic_returns


def main():
    # Determine the current script directory and build the path for sp100_prices.csv
    current_dir = os.path.dirname(__file__)
    prices_file = os.path.abspath(os.path.join(current_dir, "..", "data", "sp100_prices.csv"))

    if not os.path.exists(prices_file):
        print(f"Error: Prices file not found at {prices_file}.")
        return

    # Load the historical prices (assuming the date column is the index)
    prices_df = pd.read_csv(prices_file, index_col=0, parse_dates=True)
    print("Historical prices loaded successfully.")

    # Compute daily returns
    daily_returns = compute_daily_returns(prices_df)
    print("Daily returns computed.")

    # Compute quarterly returns; you can adjust the period if needed (e.g., 'M' for monthly)
    quarterly_returns = compute_periodic_returns(daily_returns, period='Q')
    print("Quarterly returns computed.")

    # Construct output file paths for returns
    daily_returns_file = os.path.abspath(os.path.join(current_dir, "..", "data", "sp100_daily_returns.csv"))
    quarterly_returns_file = os.path.abspath(os.path.join(current_dir, "..", "data", "sp100_quarterly_returns.csv"))

    # Save the returns to CSV files for later use in your optimization/modeling tasks
    daily_returns.to_csv(daily_returns_file)
    quarterly_returns.to_csv(quarterly_returns_file)

    print(f"Daily returns saved to: {daily_returns_file}")
    print(f"Quarterly returns saved to: {quarterly_returns_file}")


if __name__ == '__main__':
    main()
