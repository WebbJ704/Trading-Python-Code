import Rules as rules
import Signal as Sig
import backtest as bt
import pandas as pd
import numpy as np
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def RSI_variations(df):
    low_range , high_range , window_range  , best_df , best_trades , sys_top_ten = RSI_settings(df)
    low_range , high_range , window_range , best_df , best_trades, sys_top_ten =  RSI_settings(df, window_range ,low_range , high_range )
    sys_rand = sys_top_ten.sample(n=1)
    low = sys_rand.iloc[0]['low']
    high= sys_rand.iloc[0]['high']
    window = sys_rand.iloc[0]['window']
    setting = {
                "name" : f"RSI_w{window}_H{high}_L{low}",
                "window": int(window),
                "high": int(high),
                "low": int(low)
            }
    return setting , best_df , best_trades 

def RSI_settings(df, win = [5,18] , rsi_low = [20,40], rsi_high =[60,80]):
    window = np.random.randint(win[0], win[1], size=100) # best range 7,9
    rsi_low = np.random.randint(rsi_low[0], rsi_low[1], size=100) # best range 7,9
    rsi_high = np.random.randint(rsi_high[0], rsi_high[1], size=100) # best range 7,9

    RSI_settings = []
    for i in range(len(window)):
       # if (slows[i] > fasts[i]) and (fasts[i] < signals[i]):
            name = f"RSI_w{window[i]}_H{rsi_high[i]}_L{rsi_low[i]}"
            RSI_settings.append({
                "name": name,
                "window": int(window[i]),
                "high": int(rsi_high[i]),
                "low": int(rsi_low[i])
            })


    model_data, system_output , sig , best_df , best_trades, best_system = back_test_variations(df, RSI_settings)
    low_range , high_range , window_range = ML_settings(model_data)

    syst_out = pd.DataFrame(system_output)
    syst_out.set_index('setting', inplace = True)
    model_data = pd.DataFrame(model_data)
    syst_top_ten = model_data.nlargest(10,'sharp')
    best_sys_out = pd.DataFrame([best_system])
    best_sys_out.set_index('setting', inplace = True)
    print(sig)
    print(best_sys_out)
    print(syst_top_ten)

    return low_range , high_range , window_range , best_df , best_trades , syst_top_ten
    
def back_test_variations(df, RSI_settings):
    best_system = None
    best_score = 0
    sig = None
    best_trades = None
    best_df = None
    total = len(RSI_settings)
    system_output =[]
    model_data = []
    for idx, setting in enumerate(RSI_settings, 1):
            df = rules.Rules(df, rsi_window = setting['window'], dropna=False)
            df = Sig.generate_signals(df, use_RSI_trend=True, rsi_trend = setting['high'],)
            trades, signal = bt.backtest(df, rsi_trend_exit_threshold = setting['low'],use_rsi_trend_exit=True)
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
            model_data.append({     'low': setting['low'],
                                    'high': setting['high'],
                                    'window': setting['window'],
                                    'sharp': sharp,
                                })
            if trades['StrategyEquity'].iloc[-1] > best_score:
                best_system = {     'setting': setting['name'],
                                    'Sharp': sharp,
                                    'Retrun': trades['StrategyEquity'].iloc[-1],
                                    'BuyHold': trades['BuyAndHold'].iloc[-1],
                                    'Retrun Std': trades['Return'].std(),
                                    'No.of Trades': len(trades),
                                }
                df.to_csv('test')
                best_trades = trades
                best_df = df
                sig = signal[-1]
                best_score = trades['StrategyEquity'].iloc[-1]
            percent = idx / total * 100
            print(f"\rProgress: {idx}/{total} ({percent:.1f}%)",end='')
    return model_data, system_output , sig , best_df , best_trades, best_system


def ML_settings(model_data, min_score=0.25, max_retries=5):
    df_model = pd.DataFrame(model_data)

    # Drop rows with NaN in sharp, low, high, or window
    df_model = df_model.dropna(subset=['sharp', 'low', 'high', 'window'])

    #Filter out invalid RSI logic: low >= high
    df_model = df_model[df_model['low'] < df_model['high']]

    if df_model.empty:
        raise ValueError("Model data is empty after cleaning. Check RSI settings or backtest output.")

    X = df_model[['low', 'high', 'window']]
    y = df_model['sharp']

    score = 0
    retries = 0
    train_score =0

    while score < min_score and retries < max_retries:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestRegressor(n_estimators=100, max_depth = 5,ccp_alpha =0.01, random_state=retries)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        train_score =model.score(X_train,y_train)
        retries += 1

        if score < min_score:
            print(f"Retrying model training... (attempt {retries}, score={score:.3f})")

    if score < min_score:
        print(f"Warning: Final model score ({score:.3f}) is still below {min_score}")

    # Predict over parameter grid
    new_params = pd.DataFrame(product(range(20, 40), range(60, 80), range(2, 20)), columns=['low', 'high', 'window'])

    # ❗️Filter out logically invalid combinations
    new_params = new_params[new_params['low'] < new_params['high']]

    new_params['predicted_sharpe'] = model.predict(new_params)

    best = new_params.sort_values('predicted_sharpe', ascending=False).head(20)
    print("Best predicted RSI params:\n", best)
    print("Final model test score:", round(score, 4))
    print("Final model train score:", round(train_score, 4))

    low_range = [best['low'].min(), best['low'].max()]
    high_range = [best['high'].min(), best['high'].max()]
    window_range = [best['window'].min(), best['window'].max()]

    return low_range, high_range, window_range
