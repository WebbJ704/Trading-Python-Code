import Rules as rules
import Signal as Sig
import backtest as bt
import pandas as pd
import numpy as np
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def BB_variations(df):
    window_range, lower_range, upper_range, best_df , best_trades , sys_top_ten = BB_settings(df)
    window_range, lower_range, upper_range , best_df , best_trades, sys_top_ten =  BB_settings(df,window_range, lower_range, upper_range)
    sys_rand = sys_top_ten.sample(n=1)
    window = sys_rand.iloc[0]['window']
    lower = sys_rand.iloc[0]['lower']
    upper = sys_rand.iloc[0]['upper']
    setting = {
                "name": f"BB_W{window}_L{lower}_U{upper}",
                "window": window,
                "lower": lower,
                "upper": upper
            }
    return setting , best_df , best_trades 

def BB_settings(df, win = [10,30], l_std = [0.5,2.5], h_std = [0.5,2.5]):
    window = np.random.randint(win[0], win[1]+1, size=100) # best range 7,9
    lower_std = np.round(np.random.uniform(l_std[0]-0.5, l_std[1]+0.5, size=100), 1)  # round to 1 decimal
    upper_std = np.round(np.random.uniform(h_std[0]-0.5, h_std[1]+0.5, size=100), 1) # best range 12,17
    
    bb_settings = []
    for i in range(len(window)):
       # if (slows[i] > fasts[i]) and (fasts[i] < signals[i]):
            name = f"BB_W{window[i]}_l{lower_std[i]}_U{upper_std[i]}"
            bb_settings.append({
                "name": name,
                "window": int(window[i]),
                "lower std": int(lower_std[i]),
                "upper std": int(upper_std[i])
            })
    
    model_data, system_output , sig , best_df , best_trades, best_system = back_test_variations(df, bb_settings)
    fast_range , slow_range , signal_range = ML_settings(model_data)

    syst_out = pd.DataFrame(system_output)
    syst_out.set_index('setting', inplace = True)
    model_data = pd.DataFrame(model_data)
    syst_top_ten = model_data.nlargest(10,'return')
    best_sys_out = pd.DataFrame([best_system])
    best_sys_out.set_index('setting', inplace = True)
    print(sig)
    print(best_sys_out)
    print(syst_top_ten)

    return fast_range , slow_range , signal_range , best_df , best_trades , syst_top_ten
    
def back_test_variations(df, bb_settings): 
    best_system = None
    best_score = 0
    sig = None
    best_trades = None
    best_df = None
    total = len(bb_settings)
    system_output =[]
    model_data = []
    for idx, setting in enumerate(bb_settings, 1):
            df = rules.Rules(df, BB_window = setting['window'], BB_std_dev_top = setting['upper std'], BB_std_dev_bottom = setting['lower std'], dropna=False)
            df.to_csv('test')
            df = Sig.generate_signals(df, use_BB=True)
            trades, signal = bt.backtest(df,use_bb_exit = True)
            if trades.empty:
                print(f"⚠️ Skipping empty trades for setting {setting['name']}")
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
            model_data.append({     'window': setting['window'],
                                    'lower': setting['lower std'],
                                    'upper': setting['upper std'],
                                    'return': trades['StrategyEquity'].iloc[-1],
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

def ML_settings(model_data, min_score=0.25, max_retries=5):
    df_model = pd.DataFrame(model_data)

    # Drop rows with NaN in sharp, low, high, or window
    df_model = df_model.dropna(subset=['return', 'lower', 'upper', 'window'])


    X = df_model[['window', 'lower', 'upper']]
    y = df_model['return']


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

    # Predict over grid
    new_params = pd.DataFrame(product(range(10, 50), np.arange(0.5, 2.6, 0.1) , np.arange(0.5, 2.6, 0.1) ), columns=['window', 'lower', 'upper'])
    new_params['predicted_return'] = model.predict(new_params)

    best = new_params.sort_values('predicted_return', ascending=False).head(20)
    print("Best predicted BB params:\n", best)
    print("Final model test score:", round(test_score, 4))
    print("Final model train score:", round(train_score, 4))

    window_range = [best['window'].min(), best['window'].max()]
    lower_range = [best['lower'].min(), best['lower'].max()]
    upper_range = [best['upper'].min(), best['upper'].max()]

    return window_range, lower_range, upper_range