---

# Enhanced Index Fund Constructor (EIFC)

A comprehensive AI-driven solution to optimize index fund portfolios that track the **S&P 100 Index** with fewer stocks, utilizing advanced optimization techniques and insightful data analysis.

## 🚀 Overview
The EIFC project empowers investors to construct efficient and compact index funds using sophisticated Artificial Intelligence methods. By strategically selecting a subset of stocks from the S&P 100, the EIFC maintains tracking performance comparable to the full index.

## 📌 Key Features

### 📊 Data Management
- **Automated Data Collection**: Seamless download of historical stock data.
- **Robust Returns Calculation**: Compute daily and quarterly returns effortlessly.
- **Data Integrity Checks**: Ensuring accuracy and consistency across datasets.

### ⚙️ Optimization Methods
- **AMPL with Gurobi Solver**: Mathematical optimization for precise portfolio composition.
- **Genetic Algorithms (GA)**: Heuristic optimization for flexible, efficient solutions.
- **Interactive Parameters**: Easily tune parameters such as risk tolerance and the number of selected stocks (**q**).

### 📈 Advanced Analysis & Visualization
- **Interactive Dashboard**: Visualize portfolio performance, composition, and comparative analysis.
- **Performance Benchmarks**: Direct comparison against the full S&P 100.
- **Efficient Frontier & Risk Analysis**: Insight into risk-return trade-offs.
- **Optimization Progress Tracking**: Visualize and analyze convergence and stability of solutions.

### 📋 Reporting & Export
- **Performance Reports**: Generate comprehensive summaries of portfolio metrics.
- **Export Flexibility**: Save and export results in multiple formats (CSV, PDF, PNG, SVG).

## 🔧 Default Options & Expected Results

The following default parameters provide an optimal starting point for EIFC:

- **Analysis Period**:
  - **Start Date**: `2023-01-01`
  - **End Date**: `2025-03-01`
- **Number of Selected Stocks (q)**: `20`
- **GA Optimization Parameters**:
  - **Generations**: `100`
  - **Population Size**: `50`

### ✅ Expected Results

**AMPL Optimization (Default Run)**:
- **Selected Stocks**:
  - Top weighted stocks: `ACN`, `GS`, `AAPL`, `DIS`, `LLY`
- **Tracking Error**: `2.9097e-32`
- **Mean Quarterly Portfolio Return**: `5.7920%`
- **Annualized Portfolio Return**: `24.8214%`
- **Correlation with S&P 100**: `1.0000`
- **Sharpe Ratio**: `0.8596`
- **Annualized Volatility**: `8.8231%`
- **Maximum Drawdown**: `-1.7024%`

**GA Optimization (Default Run)**:
- **Selected Stocks**:
  - Top weighted stocks: `ACN`, `LIN`, `PEP`, `COST`, `JPM`
- **Tracking Error**: `1.8166e-04`
- **Mean Quarterly Portfolio Return**: `5.4130%`
- **Annualized Portfolio Return**: `23.0334%`
- **Correlation with S&P 100**: `0.9985`
- **Sharpe Ratio**: `0.7678`
- **Annualized Volatility**: `8.8905%`
- **Maximum Drawdown**: `-1.7488%`
- **Best GA Fitness (Final Generation)**: `5504.4008`

These results indicate exceptional tracking and performance, particularly using AMPL. The GA provides slightly less accurate but highly competitive performance with excellent diversification potential.(This is before tweaking any values. GA outperforms the S&P in various other scenarios.)

## 🛠 Installation

Ensure you have Python installed. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/leePettigrew/indexfund.git
cd indexfund
```

Install required Python packages:

```bash
pip install customtkinter matplotlib pandas numpy yfinance pygad amplpy
```

Ensure **AMPL** and **Gurobi** are properly installed and licensed on your system. See [AMPL website](https://ampl.com) and [Gurobi website](https://www.gurobi.com/) for setup instructions.

## 📂 Project Structure

```plaintext
EIFC/
├── data/                       # Directory for datasets
│   ├── sp100_tickers.csv       # List of S&P 100 tickers
│   ├── sp100_prices.csv        # Historical stock prices
│   ├── sp100_daily_returns.csv # Daily computed returns
│   └── sp100_quarterly_returns.csv # Quarterly computed returns
├── results/                    # Directory for optimization results
├── models/                     # AMPL optimization model files
│   └── index_fund.mod          # AMPL model definition
└── src/                        # Source code and scripts
    ├── data_collection.py      # Download historical stock data
    ├── compute_returns.py      # Compute daily & quarterly returns
    ├── optimization.py         # AMPL-based optimization routines
    ├── ga_optimization.py      # GA-based optimization routines
    ├── run_all.py              # Integrated script for all tasks
    ├── test_setup.py           # Quick environment & dependency test
    └── gui_app.py              # Comprehensive GUI for EIFC
```

## 🚦 Quick Start

### 1. Setup the Tickers CSV

Create `sp100_tickers.csv` inside the `data/` directory with at least one column named **"Ticker"**:

```csv
Ticker
AAPL
MSFT
GOOGL
...
```

### 2. Running the Project

Use the integrated script to execute the tasks:

```bash
cd src
python run_all.py
```

Follow the on-screen menu:

- Download stock data
- Compute returns
- Run optimization methods (AMPL & GA)
- View and export results

### 3. Launching the GUI

For interactive exploration and visualization:

```bash
python gui_app.py
```

Enjoy an intuitive, user-friendly interface with advanced visualization features (recommended).

##  Credit

- **Lee Pettigrew** (X20730039)

## 🎓 Project Context

This project is submitted as part of the **AI Driven Decision Making** module (H9AIDM) at the **National College of Ireland**.

- **Module Leaders**: Ade Fajemisin, Harshani Nagahamulla

---

**Developed by Lee Pettigrew – April 2025**