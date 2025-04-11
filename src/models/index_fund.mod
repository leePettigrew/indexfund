# index_fund.mod - AMPL model for optimal index fund construction

# Parameters
param n;                      # Number of available stocks (total S&P 100 stocks)
param q;                      # Number of stocks to select for our index fund (q < n)
param T;                      # Number of time periods (quarters)

# Sets
set STOCKS := 1..n;           # Set of all available stocks
set PERIODS := 1..T;          # Set of time periods

# Data parameters
param returns{STOCKS, PERIODS};  # Historical returns for each stock in each period
param sp100_returns{PERIODS};    # S&P 100 index returns for each period

# Decision variables
var select{STOCKS} binary;    # 1 if stock is selected, 0 otherwise
var weight{STOCKS} >= 0;      # Weight of each stock in the portfolio

# Portfolio return in each period
var portfolio_return{t in PERIODS} = sum{i in STOCKS} weight[i] * returns[i,t];

# Using tracking error as the objective function
minimize tracking_error:
    sum{t in PERIODS} (portfolio_return[t] - sp100_returns[t])^2;

# Constraints
subject to select_q_stocks:
    sum{i in STOCKS} select[i] = q;

subject to weights_sum_to_one:
    sum{i in STOCKS} weight[i] = 1;

subject to weight_if_selected{i in STOCKS}:
    weight[i] <= select[i];

# Optional diversification constraint - uncomment if needed
# subject to max_weight{i in STOCKS}:
#     weight[i] <= 0.3 * select[i];
