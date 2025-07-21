import Rules as rules
import Signal as Sig
import backtest as bt
import pandas as pd
import numpy as np
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def MACD_variations(df):
    fast_range , slow_range , signal_range, target_range, stop_loss_range , best_df , best_trades , syst_top_ten = MACD_settings(df)
    fast_range , slow_range , signal_range, target_range, stop_loss_range , best_df , best_trades , syst_top_ten =  MACD_settings(df,slow_range,fast_range,signal_range, target_range, stop_loss_range )
    sys_rand = syst_top_ten.sample(n=1)
    fast = sys_rand.iloc[0]['fast']
    slow = sys_rand.iloc[0]['slow']
    signal = sys_rand.iloc[0]['signal']
    target = sys_rand.iloc[0]['target']
    stop_loss = sys_rand.iloc[0]['stop loss']
    setting = {
                "name": f"MACD_f{fast}_s{slow}_sig{signal}",
                "fast": fast,
                "slow": slow,
                "signal": signal,
                "target": target,
                "stop loss": stop_loss,
            }
    return setting , best_df , best_trades 

def MACD_settings(df, sl = [21,50], f = [5,20], si = [5,25], t = [1.01,1.2], s = [0.90,0.99]):
    fasts = np.random.randint(f[0], f[1]+1, size=100) # best range 7,9
    slows = np.random.randint(sl[0], sl[1]+1, size=100) # best range 22,27
    signals = np.random.randint(si[0], si[1]+1, size=100) # best range 12,17
    target = np.round(np.random.uniform(t[0], t[1]+0.01, size=100),2)
    stop_loss = np.round(np.random.uniform(s[0]-0.01, s[1], size=100),2)
    
    macd_settings = []
    for i in range(len(fasts)):
       # if (slows[i] > fasts[i]) and (fasts[i] < signals[i]):
            name = f"MACD_f{fasts[i]}_s{slows[i]}_sig{signals[i]}"
            macd_settings.append({
                "name": name,
                "fast": int(fasts[i]),
                "slow": int(slows[i]),
                "signal": int(signals[i]),
                "target": float(target[i]),
                "stop loss": float(stop_loss[i])
            })
    model_data, system_output , sig , best_df , best_trades, best_system = back_test_variations(df, macd_settings)
    fast_range , slow_range , signal_range, target_range, stop_loss_range = ML_settings(model_data)

    syst_out = pd.DataFrame(system_output)
    syst_out.set_index('setting', inplace = True)
    model_data = pd.DataFrame(model_data)
    syst_top_ten = model_data.nlargest(10,'sharp')
    best_sys_out = pd.DataFrame([best_system])
    best_sys_out.set_index('setting', inplace = True)
    print(sig)
    print(best_sys_out)
    print(syst_top_ten)

    return fast_range , slow_range , signal_range, target_range, stop_loss_range , best_df , best_trades , syst_top_ten
    
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
            df = rules.Rules(df, EMA_short = setting['fast'], EMA_long = setting['slow'], EMA_signal = setting['signal'], dropna=False)
            df = Sig.generate_signals(df,
                                        use_ADX=False,
                                        use_RSI=False,
                                        use_EMA=False,
                                        use_MACD=True,
                                        use_ATR=False,
                                        use_volume_spike=False,
                                        use_stochastic=False,
                                        use_BB=False )
            trades, signal = bt.backtest(df,use_macd_exit = True,
                                          use_take_profit_stop_loss = True,
                                            take_profit = setting['target'],
                                            stop_loss = setting['stop loss'])
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
            model_data.append({     'fast': setting['fast'],
                                    'slow': setting['slow'],
                                    'signal': setting['signal'],
                                    'sharp': sharp,
                                    'target': setting['target'],
                                    'stop loss': setting['stop loss']
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
    X = df_model[['fast', 'slow', 'signal','target','stop loss']]
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

    # Predict over grid
    new_params = pd.DataFrame(product(range(5, 21), range(21, 51), range(5, 26),np.round(np.arange(1.01, 1.21, 0.01), 2),
        np.round(np.arange(0.90, 0.99, 0.01), 2)), columns=['fast', 'slow', 'signal','target','stop loss'])
    new_params = new_params[new_params['fast'] < new_params['slow']]
    new_params['predicted_sharpe'] = model.predict(new_params)

    best = new_params.sort_values('predicted_sharpe', ascending=False).head(20)
    print("Best predicted MACD params:\n", best)
    print("Final model test score:", round(test_score, 4))
    print("Final model train score:", round(train_score, 4))

    fast_range = [best['fast'].min(), best['fast'].max()]
    slow_range = [best['slow'].min(), best['slow'].max()]
    signal_range = [best['signal'].min(), best['signal'].max()]
    target_range = [best['target'].min(), best['target'].max()]
    stop_loss_range = [best['stop loss'].min(), best['stop loss'].max()]

    return fast_range, slow_range, signal_range , target_range, stop_loss_range