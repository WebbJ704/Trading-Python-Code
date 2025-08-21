import yfinance as yf
import pandas as pd 
import os
from datetime import date
from scipy.optimize import differential_evolution
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
        data = yf.download(ticker, period="max", interval="1d", auto_adjust=True)
        data.reset_index(inplace=True) 
        df = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["date", "Open", "High", "Low", "Close", "Volume"]
        df["date"] = pd.to_datetime(df["date"]) 
        df = df.set_index("date")
        df.to_csv(filename)
        df = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        return df

# ----------------- Feature Engineering / Signal Generation -----------------

def Rules(df, **kwargs):
    EMA_short   = kwargs.get("EMA_short", 12)
    EMA_long    = kwargs.get("EMA_long", 26)
    EMA_signal  = kwargs.get("EMA_signal", 9)
    BB_window   = kwargs.get("BB_window", 35)
    BB_std_dev_top  = kwargs.get("BB_std_dev_top", 1.5)
    BB_std_dev_bottom  = kwargs.get("BB_std_dev_bottom", 0.5)

    # moving averages
    df['EMA_short'] = df['Close'].ewm(span=EMA_short, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=EMA_long, adjust=False).mean()

    # moving average crossover divergence 
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['MACD_Signal'] = df['MACD'].ewm(span=EMA_signal, adjust=False).mean()

    #Bollinger bands exponetaly weighted
    df['BB_Middle'] = df['Close'].ewm(span=BB_window, adjust=False).mean()
    df['BB_StdDev'] = df['Close'].ewm(span=BB_window, adjust=False).std()
    df['BB_Upper'] = df['BB_Middle'] + BB_std_dev_top * df['BB_StdDev']
    df['BB_Lower'] = df['BB_Middle'] - BB_std_dev_bottom * df['BB_StdDev']

    if kwargs.get("dropna", False):
        df.dropna(inplace=True)

    return df 


def generate_signals(df, **kwargs):
    # Define indicator toggles with default values
    use_MACD = kwargs.get("use_MACD", False)
    use_BB = kwargs.get("use_BB", False)

    df['Signal'] = 0

    for i in range(1, len(df)):
        conditions = []

        # Each block is optional based on kwargs
        if use_MACD:
            conditions.append(df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i])

        if use_BB:
            conditions.append(df['Close'].iloc[i] < df['BB_Lower'].iloc[i])


        # Only assign signal if ALL enabled conditions are met
        if all(conditions):
            df.loc[df.index[i], 'Signal'] = 1

    return df

# ------------------------------------------

# ----------------- Strategy Backtesting -----------------

def backtest(df, **kwargs):
    trades = []
    Signal = []
    df['Sell Signal'] = 0
    position = 0
    entry_price = 0
    entry_date = None

    # Default exit condition toggles
    use_macd_exit = kwargs.get("use_MACD", False)

    use_bb_exit = kwargs.get("use_BB", False)

    for i in range(1, len(df)-1):
        if df['Signal'].iloc[i] == 1 and position == 0:
            position = 1
            entry_price = df['Close'].iloc[i]
            entry_date = df.index[i]
            Signal.append(f'Buy - {entry_date, entry_price}')

        elif position == 1:
            exit_conditions = []

            if use_macd_exit:
                exit_conditions.append(df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i])
         
            if use_bb_exit:
                exit_conditions.append(df['Close'].iloc[i] > df['BB_Upper'].iloc[i])


            # Exit trade if ALL of the enabled conditions are True
            if any(exit_conditions):
                exit_price = df['Close'].iloc[i]
                exit_date = df.index[i+1]
                trade_return = (exit_price - entry_price) / entry_price
                trades.append({
                    'EntryDate': entry_date,
                    'ExitDate': exit_date,
                    'Return': trade_return
                })
                Signal.append(f'Sell - {exit_date, exit_price}')
                position = 0
                entry_date = None
                df.loc[i, "Sell Signal"] = 1 

    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.set_index('ExitDate', inplace=True)
        first_entry_date = trades_df['EntryDate'].iloc[0]
        buy_hold_start_price = df.loc[first_entry_date, 'Close']
        trades_df['BuyAndHold'] = df.loc[trades_df.index, 'Close'].values / buy_hold_start_price
        trades_df['StrategyEquity'] = (1 + trades_df['Return']).cumprod()
    else:
        trades_df = pd.DataFrame(columns=['EntryDate', 'ExitDate', 'Return', 'BuyAndHold'])

    return trades_df, Signal

# ------------------------------------------

# ----------------- strategy Pipeline -----------------

def strat_pipeline(df, tickers, params, **kwargs):
    use_MACD = kwargs.get("use_MACD", False)
    use_BB = kwargs.get("use_BB", False)
    trades_df = {}
    signals_df = {}
    for ticker in tickers:
        ticker_df = df[ticker].copy()
        fast, slow, signal, window, lower_std, upper_std = params[ticker]
        df_tick = Rules(ticker_df, EMA_short = fast, EMA_long = slow, EMA_signal = signal, 
                            BB_window = window, BB_std_dev_top = upper_std, BB_std_dev_bottom = lower_std, dropna=False)
        df_tick = generate_signals(df_tick,use_BB = use_BB, use_MACD = use_MACD)
        trades_df_tick, signals = backtest(df_tick,use_BB = use_BB, use_MACD = use_MACD)
        trades_df[ticker] = trades_df_tick
        signals_df[ticker] = signals
    return trades_df , signals

# ----------------- Performance of different variations -----------------    
def variation_perfromace_plot(df,tickers,**kwargs):
    use_MACD = kwargs.get("use_MACD", False)
    use_BB = kwargs.get("use_BB", False)
    trade_means = {}
    for ticker in tickers:
        start = df[ticker].index.min()
        end   = df[ticker].index.max()
        total_days  = (end - start).days
        time_years  = total_days / 365.25
        total_return = []
        total_return_std = []
        means = []
        std= []
        ticker_df = df[ticker].copy()
        for _ in tqdm(range(500)):
            fast = np.random.randint(5, 20, size=1) 
            slow = np.random.randint(21, 50, size=1) 
            signal = np.random.randint(5, 25, size=1) 
            window = np.random.randint(10, 30, size=1) 
            lower_std = np.round(np.random.uniform(0.5, 2, size=1), 1) 
            upper_std = np.round(np.random.uniform(0.5, 2, size=1), 1)  
            df_tick = Rules(ticker_df, EMA_short = fast, EMA_long = slow, EMA_signal = signal, 
                                BB_window = window, BB_std_dev_top = upper_std, BB_std_dev_bottom = lower_std, dropna=False)
            df_tick = generate_signals(df_tick,use_BB = use_BB, use_MACD = use_MACD)
            trades_df, signals = backtest(df_tick,use_BB = use_BB, use_MACD = use_MACD)

            if trades_df.empty:
                continue
            else:
                returns_mean = trades_df['Return'].mean()
                retrun_std =  trades_df['Return'].std()
                total = trades_df['StrategyEquity'].iloc[-1] - 1 
                total_std = trades_df['StrategyEquity'].std()

            means.append(returns_mean)
            std.append(retrun_std)
            total_return.append(total)
            total_return_std.append(total_std)
        means = np.array(means, dtype=float)
        std = np.array(std, dtype=float)
        total_return = np.array(total_return, dtype=float)
        total_return_std = np.array(total_return_std, dtype=float)

        trade_means[ticker] = means

        plt.scatter(std, means, alpha = 0.4, edgecolor = 'black')
        plt.title(f'Risk vs. reward of Rule variations for {ticker}')
        plt.grid()
        plt.xlabel('Volatility Per Trade')
        plt.ylabel('Mean return Per Trade')
        plt.show()
        plt.scatter(total_return_std, total_return, alpha = 0.4, edgecolor = 'black')
        plt.title(f'Risk vs. reward of Rule variations for {ticker} over {time_years:.1f} years')
        plt.grid()
        plt.xlabel('Overall Volatility')
        plt.ylabel('Overall Return')
        plt.show()
    return trade_means

# ----------------- variation Performance Evaluation with scipy optimise -----------------    
# ---- Objective Function (Sharpe ratio with return penalty) ----
def objective_sharpe(params, df, use_MACD=False, use_BB=False, return_threshold=0.10):
    fast, slow, signal, window, lower_std, upper_std = params
    
    # Skip impossible combos (MACD constraint)
    if fast >= slow or lower_std <= 0 or upper_std <= 0:
        return 1e6

    # Convert only when applying to trading rules
    fast_i = int(round(fast))
    slow_i = int(round(slow))
    signal_i = int(round(signal))
    window_i = int(round(window))

    df_copy = df.copy()
    df_copy = Rules(
        df_copy,
        EMA_short=fast_i,
        EMA_long=slow_i,
        EMA_signal=signal_i,
        BB_window=window_i,
        BB_std_dev_top=upper_std,
        BB_std_dev_bottom=lower_std,
        dropna=True
    )
    df_copy = generate_signals(df_copy, use_MACD=use_MACD, use_BB=use_BB)
    trades_df, _ = backtest(df_copy, use_MACD=use_MACD, use_BB=use_BB)

    if trades_df.empty:
        return 1e6
    
    return_mean = trades_df['Return'].mean()
    return_std =  trades_df['Return'].std()
    sharpe = return_mean/return_std

    # # Penalize if return is below threshold
    # if return_mean < return_threshold:
    #     penalty = (return_threshold - return_mean) * 100
    #     return 1e6 + penalty

    return -sharpe

def scipy_optimise(df, tickers, trade_means, **kwargs):
    use_MACD = kwargs.get("use_MACD", False)
    use_BB = kwargs.get("use_BB", False)
    params = {}

    bounds = [
        (5, 20),    # fast
        (21, 50),   # slow
        (5, 25),    # signal
        (10, 30),   # window
        (0.5, 1.5), # lower_std
        (0.5, 2.0)  # upper_std
    ]

    for ticker in tickers:
        return_threshold = trade_means[ticker].mean() #threshhold is calculated by find the mean return per trade from variaition plot + 1 std
        start = df[ticker].index.min()
        end   = df[ticker].index.max()
        total_days  = (end - start).days
        time_years  = total_days / 365.25
        print(f"\nOptimizing {ticker}...")

        # Create tqdm progress bar
        pbar = tqdm(total=100, desc=f"{ticker} progress", ncols=80)
        def tqdm_callback(xk, convergence):
            pbar.update(1)
            return False  # return True to stop early

        result = differential_evolution(
            objective_sharpe,
            bounds=bounds,
            args=(df[ticker], use_MACD, use_BB, return_threshold),
            maxiter=100,
            polish=True,
            callback=tqdm_callback
        )
        pbar.close()
        best_params = result.x
        best_sharpe = -result.fun
        params[ticker] = best_params
        print(f"Optimized parameters for {ticker}: {best_params}")
        print(f"Max Sharpe ratio per trade over {time_years:.1f} years for {ticker} with return >= {return_threshold}: {best_sharpe:.4f}")
    return params

# ----------------- BB_MACD Variations Performance Evaluation and optimisation with Optuna -----------------    

def evaluate_pipeline(df, EMA_short, EMA_long, EMA_signal, BB_window, BB_std_dev_top,BB_std_dev_bottom):

    df = Rules(df, EMA_short = EMA_short, EMA_long = EMA_long, EMA_signal = EMA_signal, 
                    BB_window = BB_window, BB_std_dev_top = BB_std_dev_top, BB_std_dev_bottom = BB_std_dev_bottom, dropna=False)
    df = generate_signals(df, use_BB=True, use_MACD=True)
    trades_data, signals = backtest(df, use_BB=True, use_MACD=True)

    returns = trades_data["Return"]
    returns_mean = returns.mean()
    std = returns.std()

    sharpe = returns_mean / std 

    return sharpe, returns


def objective(trial, df):
    EMA_short = trial.suggest_int('EMA_short', 5, 20)
    EMA_long = trial.suggest_int('EMA_long', 21, 50)
    EMA_signal = trial.suggest_int('EMA_signal', 5, 25)
    BB_window = trial.suggest_int('BB_window', 10, 30)
    BB_std_dev_top = trial.suggest_float('BB_std_dev_top',0.5,2)
    BB_std_dev_bottom = trial.suggest_float('BB_std_dev_bottom',0.5,2)

    sharpe, port_returns = evaluate_pipeline(df.copy(),  EMA_short, EMA_long, EMA_signal, BB_window, BB_std_dev_top, BB_std_dev_bottom)

    trial.set_user_attr("portfolio_returns", port_returns)
    
    return sharpe
