import yfinance as yf
import pandas as pd 
import os
from datetime import date
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

# ----------------- Data Acquisition & Preprocessing -----------------
def fetch_yf_data(ticker, start = '2020-09-18', end = date.today().strftime("%Y-%m-%d")):
    filename = f'{ticker}_data_for_book.csv'
    
    if os.path.exists(filename):
        print(f"{filename} already exists. Loading from disk...")
        df = pd.read_csv(filename)
        df.reset_index(inplace=True) 
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        return df
    else:
        print(f"Downloading data for {ticker}...")
        data = yf.download(ticker, auto_adjust=False) 
        data.reset_index(inplace=True) 
        df = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["date", "Open", "High", "Low", "Close", "Volume"]
        df["date"] = pd.to_datetime(df["date"]) 
        df = df.set_index("date")
        df.to_csv(filename)
        df = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        return df

# ----------------- Feature Engineering / Signal Generation -----------------

def stochiastic_oscilaotr(df, **kwargs):
    k_period    = kwargs.get("k_period", 14)
    d_period    = kwargs.get("d_period", 3)
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    df['%K'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    return df

def signal_genorator(df, **kwargs):
    lower_k_threshold = kwargs.get("lower_k_threshold", 20)
    for i in range(1,len(df)):
     if (df['%K'].iloc[i] > df['%D'].iloc[i] and
         df['%K'].iloc[i] < lower_k_threshold and
         df['%D'].iloc[i] < lower_k_threshold):

         df.loc[df.index[i], 'Signal'] = 1
    return df

# ------------------------------------------

# ----------------- Strategy Backtesting -----------------

def backtest(df,**kwargs):
    stochastic_upper = kwargs.get("stochastic_upper", 80)
    trades = []
    position = 0
    entry_price = 0
    entry_date = None
    for i in range(1, len(df)-1):
        if df['Signal'].iloc[i-1] == 1 and position == 0:
            position = 1
            entry_price = df['Open'].iloc[i]
            entry_date = df.index[i]

        elif position == 1:
            if (df['%K'].iloc[i] < df['%D'].iloc[i] and
                    df['%K'].iloc[i] > stochastic_upper and
                    df['%D'].iloc[i] > stochastic_upper):
                exit_price = df['Open'].iloc[i+1]
                exit_date = df.index[i+1]
                trade_return = (exit_price - entry_price) / entry_price
                trades.append({
                    'EntryDate': entry_date,
                    'ExitDate': exit_date,
                    'Return': trade_return
                })
                position = 0
                entry_date = None
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.set_index('ExitDate', inplace=True)
        first_entry_date = trades_df['EntryDate'].iloc[0]
        sharpe = trades_df["Return"].mean() / trades_df['Return'].std()
        buy_hold_start_price = df.loc[first_entry_date, 'Close']
        trades_df['BuyAndHold'] = df.loc[trades_df.index, 'Close'].values / buy_hold_start_price
        trades_df['StrategyEquity'] = (1 + trades_df['Return']).cumprod()
    else:
        trades_df = pd.DataFrame(columns=['EntryDate', 'ExitDate', 'Return', 'BuyAndHold',])
    return trades_df, sharpe

# ------------------------------------------

# ----------------- Performance Evaluation / Statistical Analysis -----------------

def bootstrap(trades,n_simulations=1000):
    bootstrap_results = [] 
    for i in range(1, n_simulations):
        sample = trades['Return'].sample(len(trades), replace = True, random_state = i)
        sample_mean = sample.mean()
        sample_std = sample.std()
        sharp = sample_mean/sample_std
        bootstrap_results.append({"c0_means":sample_mean,
                        "c0_dev":sample_std,
                        "sharp":sharp})
    return bootstrap_results


def T_test(trades_df, df_full, n_simulations=1000):
    if trades_df.empty or 'Return' not in trades_df.columns:
        print("Invalid or empty trades DataFrame.")
        return None

    # Calculate strategy Sharpe ratio distribution via bootstrap
    strat_sharpes = []
    for i in range(n_simulations):
        sample = trades_df['Return'].sample(len(trades_df), replace=True, random_state=i)
        if sample.std() != 0:
            strat_sharpes.append(sample.mean() / sample.std())

    # Calculate buy-and-hold returns over same period
    first_entry = trades_df['EntryDate'].iloc[0]
    last_exit = trades_df.index[-1]
    bh_returns = df_full.loc[first_entry:last_exit, 'Close'].pct_change().dropna()

    # Bootstrap Sharpe ratios for buy-and-hold
    bh_sharpes = []
    for i in range(n_simulations):
        bh_sample = bh_returns.sample(len(bh_returns), replace=True, random_state=i)
        if bh_sample.std() != 0:
            bh_sharpes.append(bh_sample.mean() / bh_sample.std())

    # Two-sample t-test on Sharpe distributions
    t_stat, p_value = stats.ttest_ind(strat_sharpes, bh_sharpes, equal_var=False)

    print(f"Strategy Sharpe (mean): {pd.Series(strat_sharpes).mean():.4f}")
    print(f"Buy & Hold Sharpe (mean): {pd.Series(bh_sharpes).mean():.4f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Result: Significant difference between strategy and buy-and-hold Sharpe ratios (p < 0.05).")
    else:
        print("Result: No significant difference in Sharpe ratios (p >= 0.05).")

    strat_sharp_mean = pd.Series(strat_sharpes).mean()

    return t_stat, p_value, strat_sharp_mean

# ------------------------------------------


# ----------------- Bayesian Optimizations -----------------

def evaluate_pipeline(df, k_period, d_period, lower_k_threshold, stochastic_upper):
    try:
        df = stochiastic_oscilaotr(df, k_period=k_period, d_period=d_period)
        df = signal_genorator(df, lower_k_threshold=lower_k_threshold)
        trades_df, sharpe = backtest(df, stochastic_upper=stochastic_upper)  
        return sharpe if not np.isnan(sharpe) else -1.0
    except Exception as e:
        print(f"Error in evaluate_pipeline: {e}")
        return -1.0

def objective(trial, train_df, test_df):
    k_period = trial.suggest_int('k_period', 5, 30)
    d_period = trial.suggest_int('d_period', 2, 10)
    lower_k_threshold = trial.suggest_float('lower_k_threshold', 10, 30)
    stochastic_upper = trial.suggest_float('stochastic_upper', 70, 90)

    train_sharpe = evaluate_pipeline(train_df.copy(), k_period, d_period, lower_k_threshold, stochastic_upper)
    test_sharpe = evaluate_pipeline(test_df.copy(), k_period, d_period, lower_k_threshold, stochastic_upper)

    trial.set_user_attr("train_sharpe", train_sharpe)
    trial.set_user_attr("test_sharpe", test_sharpe)

    return test_sharpe

# ------------------------------------------


# ----------------- Random Forest Optimizations -----------------

def evaluate_pipe(df, k_period, d_period, lower_k_threshold, stochastic_upper):
    # Your existing function that returns sharpe (ignore mse here for now)
    try:
        df = stochiastic_oscilaotr(df, k_period=k_period, d_period=d_period)
        df = signal_genorator(df, lower_k_threshold=lower_k_threshold)
        trades_df, sharpe = backtest(df, stochastic_upper=stochastic_upper)
        return sharpe if not np.isnan(sharpe) else -1.0
    except Exception as e:
        print(f"Error in evaluate_pipeline: {e}")
        return -1.0
    
param_results = []
def obj(trial, train_df):
    # Sample hyperparameters
    k_period = trial.suggest_int('k_period', 5, 30)
    d_period = trial.suggest_int('d_period', 2, 10)
    lower_k_threshold = trial.suggest_float('lower_k_threshold', 10, 30)
    stochastic_upper = trial.suggest_float('stochastic_upper', 70, 90)

    sharpe = evaluate_pipe(train_df.copy(), k_period, d_period, lower_k_threshold, stochastic_upper)
    
    # Store the parameters and score for training the RF model later
    param_results.append({
        'k_period': k_period,
        'd_period': d_period,
        'lower_k_threshold': lower_k_threshold,
        'stochastic_upper': stochastic_upper,
        'sharpe': sharpe
    })

    return  param_results

# ----------------- Data Visualization -----------------
def get_bins(data):
    n = len(data)
    return min(max(int(np.sqrt(n)), 10), 100) 

def plot_bootstrap(results):
    results = pd.DataFrame(results)
    fig3, ax3 = plt.subplots(1, 3, figsize=(20, 4))
    sns.histplot(results['c0_means'], bins=get_bins(results['c0_means']), kde=True, ax=ax3[0])
    ax3[0].set_title('Distribution of system return Means')
    sns.histplot(results['c0_dev'], bins=get_bins(results['c0_dev']), kde=True, ax=ax3[1])
    ax3[1].set_title('Distribution of system retrun Standard Deviations')
    sns.histplot(results['sharp'], bins=get_bins(results['sharp']), kde=True, ax=ax3[2])
    ax3[2].axvline(np.mean(results['sharp'])+ 2 * np.std(results['sharp']), linestyle='--', label='+2σ')
    ax3[2].axvline(np.mean(results['sharp']) - 2 * np.std(results['sharp']), linestyle='--', label='-2σ')
    ax3[2].set_title('Dsitribution of system Sharp Ratios')
    ax3[2].legend()
    plt.tight_layout()
    plt.show()

def cumulative_return(trades):
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot the strategy and buy & hold
    ax.plot(trades.index, trades['StrategyEquity'], label='Strategy')
    ax.plot(trades.index, trades['BuyAndHold'], label='Buy & Hold', linestyle='--')
    # Format the x-axis as datetime
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.tick_params(axis='x', rotation=45)
    # Add titles and labels
    ax.set_title('Cumulative Returns Comparison')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()





