import os
import yfinance as yf
import pandas as pd


def download_stock_data(tickers, start_date, end_date):
    """
    Downloads the historical adjusted closing prices for a list of tickers
    between start_date and end_date, ensuring the data is stored as a 1D Series.

    Parameters:
        tickers (list): A list of stock ticker symbols.
        start_date (str): The start date for downloading data (format: 'YYYY-MM-DD').
        end_date (str): The end date for downloading data (format: 'YYYY-MM-DD').

    Returns:
        pd.DataFrame: A DataFrame with dates as the index and ticker symbols as columns.
    """
    data = {}
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        try:
            # Download data with auto_adjust=False so that we get the "Adj Close" column intact
            ticker_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
            # Check if data is empty
            if ticker_data.empty:
                print(f"Warning: No data found for {ticker}. Skipping ticker.")
                continue

            # Ensure the "Adj Close" column exists
            if "Adj Close" not in ticker_data.columns:
                print(f"Warning: 'Adj Close' column not found for {ticker}. Skipping ticker.")
                continue

            # Use squeeze() to convert the column to a 1D Series if it's not already
            adj_close_series = ticker_data["Adj Close"].squeeze()
            # Verify that the result is one-dimensional
            if adj_close_series.ndim != 1:
                print(f"Error: Data for {ticker} is not 1-dimensional after squeeze(). Skipping ticker.")
                continue

            data[ticker] = adj_close_series
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")

    # Combine all Series into a DataFrame (aligned by index)
    if not data:
        print("No valid data found for any tickers.")
        return pd.DataFrame()

    df = pd.concat(data, axis=1)
    return df


def main():
    # Determine the directory where data_collection.py is located
    current_dir = os.path.dirname(__file__)

    # Construct absolute paths to the tickers CSV file and the output CSV file
    tickers_file = os.path.abspath(os.path.join(current_dir, "..", "data", "sp100_tickers.csv"))
    output_prices_file = os.path.abspath(os.path.join(current_dir, "..", "data", "sp100_prices.csv"))

    print(f"Looking for tickers file at: {tickers_file}")
    if not os.path.exists(tickers_file):
        print(f"Error: Tickers file not found at {tickers_file}. Please create it with a column 'Ticker'.")
        return

    # Read the tickers CSV file
    tickers_df = pd.read_csv(tickers_file)
    if "Ticker" not in tickers_df.columns:
        print("Error: CSV file must have a column named 'Ticker'.")
        return

    # Create a list of unique tickers (ignoring any NaN values)
    tickers = tickers_df["Ticker"].dropna().unique().tolist()
    print(f"Found {len(tickers)} tickers.")

    # Define the date range for the historical data
    start_date = "2023-01-01"
    end_date = "2025-03-01"

    # Download the historical data using the list of tickers
    prices_df = download_stock_data(tickers, start_date, end_date)

    # Verify that data was downloaded
    if prices_df.empty:
        print("Error: No stock data was downloaded. Exiting.")
        return

    # Save the DataFrame with historical prices to a CSV file
    prices_df.to_csv(output_prices_file)
    print(f"Historical price data successfully saved to {output_prices_file}")


if __name__ == "__main__":
    main()
