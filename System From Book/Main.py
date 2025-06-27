import data_download as dataDownlaod
import Rules as rules
import Signal as Sig
import backtest as bt
import pandas as pd
import matplotlib.pyplot as plt
import Plotting as pl
import numpy as np
import itertools
import time
# ----------------- CONFIG -----------------
SYMBOL = 'AAPL'
# ------------------------------------------
if __name__ == "__main__" :
    df = dataDownlaod.fetch_yf_data(SYMBOL,'2020 -09-18')
    plt.plot(df['Adj Close'])
    plt.show()
    fasts = np.random.randint(7, 9, size=100)
    slows = np.random.randint(22, 27, size=100)
    signals = np.random.randint(12, 17, size=100)

    """For Specific Settings"""
    # fasts = [8,9,11,13,11]
    # slows = [24,38,35,58,29]
    # signals = [13,10,8,8,8]

    """Finds the best for given range of settings"""
    fast_range = range(2, 20)    # 5 to 12
    slow_range = range(21, 60)   # 26 to 40
    signal_range = range(2, 18)  # 8 to 16
    all_variations = [
        (f, s, sig)
        for f, s, sig in itertools.product(fast_range, slow_range, signal_range)
        if f < s
    ]
    print(f"Total combinations: {len(all_variations)}")  # ~720 unique combos
    # Example to get separate lists
    # fasts = [x[0] for x in all_variations]
    # slows = [x[1] for x in all_variations]
    # signals = [x[2] for x in all_variations]
    
    macd_settings = []
    for i in range(len(fasts)):
       # if (slows[i] > fasts[i]) and (fasts[i] < signals[i]):
            name = f"MACD_f{fasts[i]}_s{slows[i]}_sig{signals[i]}"
            macd_settings.append({
                "name": name,
                "fast": int(fasts[i]),
                "slow": int(slows[i]),
                "signal": int(signals[i])
            })
    system_output =[]
    best_system = None
    sig = None
    best_trades = None
    best_df = None
    best_score = -np.inf
    start_time = time.time()
    total = len(macd_settings)

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
        trades, signal = bt.backtest(df)
        trades['StrategyEquity'] = (1 + trades['Return']).cumprod()
        sharp = trades["Return"].mean() / trades['Return'].std()
        system_output.append({  'setting': setting['name'],
                                'Sharp': sharp,
                                'Retrun': trades['StrategyEquity'].iloc[-1],
                                'BuyHold': trades['BuyAndHold'].iloc[-1],
                                'Retrun Std': trades['Return'].std(),
                                'No.of Trades': len(trades),
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
        elapsed = time.time() - start_time
        percent = idx / total * 100
        print(f"\rProgress: {idx}/{total} ({percent:.1f}%)",end='')

    pl.plot(best_df, best_trades)
    syst_out = pd.DataFrame(system_output)
    syst_out.set_index('setting', inplace = True)
    best_sys_out = pd.DataFrame([best_system])
    best_sys_out.set_index('setting', inplace = True)
    print(syst_out)
    print(sig)
    print(best_sys_out)