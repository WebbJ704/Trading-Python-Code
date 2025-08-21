import Rules as rules
import Signal as Sig
import backtest as bt
import pandas as pd
import numpy as np
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from random import sample

def MACD_variations(df):
    window_range, lower_range, upper_range, fast_range , slow_range , signal_range , best_df , best_trades , sys_top_ten = MACD_settings(df)
    window_range, lower_range, upper_range, fast_range , slow_range , signal_range , best_df , best_trades, sys_top_ten =  MACD_settings(df,slow_range,fast_range,signal_range, window_range, lower_range, upper_range)
    sys_rand = sys_top_ten.sample(n=1)
    fast = sys_rand.iloc[0]['fast']
    slow = sys_rand.iloc[0]['slow']
    signal = sys_rand.iloc[0]['signal']
    window = sys_rand.iloc[0]['window']
    lower = sys_rand.iloc[0]['lower']
    upper = sys_rand.iloc[0]['upper']
    setting = {
                "name": f"MACD_f{fast}_s{slow}_sig{signal}",
                "fast": fast,
                "slow": slow,
                "signal": signal,
                "window": window,
                "lower": lower,
                "upper": upper
            }
    return setting , best_df , best_trades 

def MACD_settings(df, sl = [21,50], f = [5,20], si = [5,25], win = [10,30], l_std = [0.5,1.5], h_std = [0.5,2]):
    fasts = np.random.randint(f[0], f[1]+1, size=100) # best range 7,9
    slows = np.random.randint(sl[0], sl[1]+1, size=100) # best range 22,27
    signals = np.random.randint(si[0], si[1]+1, size=100) # best range 12,17
    window = np.random.randint(win[0], win[1]+1, size=100) 
    lower_std = np.round(np.random.uniform(l_std[0]-0.2, l_std[1], size=100), 1)  # round to 1 decimal
    upper_std = np.round(np.random.uniform(h_std[0], h_std[1]+0.2, size=100), 1) 

    macd_settings = []
    for i in range(len(fasts)):
       # if (slows[i] > fasts[i]) and (fasts[i] < signals[i]):
            name = f"MACD_f{fasts[i]}_s{slows[i]}_sig{signals[i]}"
            macd_settings.append({
                "name": name,
                "fast": int(fasts[i]),
                "slow": int(slows[i]),
                "signal": int(signals[i]),
                "window": int(window[i]),
                "lower std": float(lower_std[i]),
                "upper std": float(upper_std[i])
            })
    
    model_data, system_output , sig , best_df , best_trades, best_system = back_test_variations(df, macd_settings)
    print("Backtest complete")
    fast_range , slow_range , signal_range, window_range, lower_range, upper_range = ML_settings(model_data)

    syst_out = pd.DataFrame(system_output)
    syst_out.set_index('setting', inplace = True)
    model_data = pd.DataFrame(model_data)
    syst_top_ten = model_data.nlargest(10,'sharp')
    best_sys_out = pd.DataFrame([best_system])
    best_sys_out.set_index('setting', inplace = True)
    print(sig)
    print(best_sys_out)
    print(syst_top_ten)

    return window_range, lower_range, upper_range, fast_range , slow_range , signal_range , best_df , best_trades , syst_top_ten
    
def back_test_variations(df, macd_settings): 
    best_system = None
    best_score = 0
    sig = None
    best_trades = None
    best_df = None
    total = len(macd_settings)
    system_output =[]
    model_data = []
    for idx, setting in enumerate(macd_settings, 1):
            df = rules.Rules(df, EMA_short = setting['fast'], EMA_long = setting['slow'], EMA_signal = setting['signal'], 
                                 BB_window = setting['window'], BB_std_dev_top = setting['upper std'], BB_std_dev_bottom = setting['lower std'], dropna=False)
            df = Sig.generate_signals(df,
                                        use_ADX=False,
                                        use_RSI=False,
                                        use_EMA=False,
                                        use_MACD=True,
                                        use_ATR=False,
                                        use_volume_spike=False,
                                        use_stochastic=False,
                                        use_BB=True )
            trades, signal = bt.backtest(df,use_macd_exit = True, use_bb_exit = True)
            if trades.empty:
                print(f"Skipping empty trades for setting {setting['name']}")
                continue
            trades['StrategyEquity'] = (1 + trades['Return']).cumprod()
            sharp = trades["Return"].mean() / trades['Return'].std()
            system_output.append({  'setting': setting['name'],
                                    'Sharp': sharp,
                                    'Retrun': trades['StrategyEquity'].iloc[-1],
                                    'BuyHold': trades['BuyAndHold'].iloc[-1],
                                    'Retrun Std': trades['Return'].std(),
                                    'No.of Trades': len(trades),
                                })
            model_data.append({     'fast': setting['fast'],
                                    'slow': setting['slow'],
                                    'signal': setting['signal'],
                                    'window': setting['window'],
                                    'lower': setting['lower std'],
                                    'upper': setting['upper std'],
                                    'sharp': sharp,
                                })
            if sharp > best_score:
                best_system = {     'setting': setting['name'],
                                    'Sharp': sharp,
                                    'Retrun': trades['StrategyEquity'].iloc[-1],
                                    'BuyHold': trades['BuyAndHold'].iloc[-1],
                                    'Retrun Std': trades['Return'].std(),
                                    'No.of Trades': len(trades),
                                }
                best_trades = trades
                best_df = df
                sig = signal[-1]
                best_score = sharp
            percent = idx / total * 100
            print(f"\rProgress: {idx}/{total} ({percent:.1f}%)",end='')
    return model_data, system_output , sig , best_df , best_trades, best_system

def ML_settings(model_data, min_score=0.5, max_retries=20):
    print("Starting ML_settings()")
    df_model = pd.DataFrame(model_data)

    # Drop rows with NaN in sharp, low, high, or window
    df_model = df_model.dropna(subset=['sharp', 'lower', 'upper', 'window','fast','slow','signal'])

    X = df_model[['fast', 'slow', 'signal','window','lower','upper']]
    y = df_model['sharp']

    test_score = 0
    retries = 0
    train_score = 0

    while test_score < min_score and retries < max_retries:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestRegressor(n_estimators=100, max_depth = 5, random_state=retries)  # vary random_state
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        retries += 1

        if test_score < min_score:
            print(f"Retrying model training... (attempt {retries}, score={test_score:.3f})")

    if test_score < min_score:
        print(f"Warning: Final model score ({test_score:.3f}) is still below {min_score}")


    # Create the full product
    full_grid = list(product(
        range(5, 21),           # fast
        range(21, 51),          # slow
        range(5, 26),           # signal
        range(10, 50),          # window
        np.round(np.arange(0.5, 2.6, 0.1), 1),  # lower
        np.round(np.arange(0.5, 2.6, 0.1), 1)   # upper
    ))

    # Sample a manageable number
    grid_sample = sample(full_grid, 5000)  # try 5k to 10k for now

    # Convert to DataFrame
    new_params = pd.DataFrame(grid_sample, columns=['fast', 'slow', 'signal','window','lower','upper'])
    new_params = new_params[new_params['fast'] < new_params['slow']]
    new_params['predicted_sharpe'] = model.predict(new_params)

    best = new_params.sort_values('predicted_sharpe', ascending=False).head(20)
    print("Best predicted MACD params:\n", best)
    print("Final model test score:", round(test_score, 4))
    print("Final model train score:", round(train_score, 4))

    fast_range = [best['fast'].min(), best['fast'].max()]
    slow_range = [best['slow'].min(), best['slow'].max()]
    signal_range = [best['signal'].min(), best['signal'].max()]
    window_range = [best['window'].min(), best['window'].max()]
    lower_range = [best['lower'].min(), best['lower'].max()]
    upper_range = [best['upper'].min(), best['upper'].max()]

    return fast_range, slow_range, signal_range, window_range, lower_range, upper_range