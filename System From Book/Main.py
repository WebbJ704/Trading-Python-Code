import data_download as dataDownlaod
import Rules as rules
import Signal as Sig
import backtest as bt
import pandas as pd
import matplotlib.pyplot as plt
# ----------------- CONFIG -----------------
SYMBOL = 'AAPL'
# ------------------------------------------
if __name__ == "__main__" :
    df = dataDownlaod.fetch_yf_data(SYMBOL,'2020 -09-18')
    plt.plot(df['Close'])
    plt.show()
    macd_settings = [
        {"name": "MACD_1", "fast": 8, "slow": 21, "signal": 9},
        {"name": "MACD_2", "fast": 12, "slow": 26, "signal": 9},
        {"name": "MACD_3", "fast": 5, "slow": 35, "signal": 5},
        {"name": "MACD_4", "fast": 20, "slow": 50, "signal": 18},
    ]
    df_macd = pd.DataFrame(macd_settings)
    df_macd.set_index('name', inplace=True)
    print(df_macd)
    system_output =[]
    for setting in macd_settings:
        df = rules.Rules(df, EMA_short = setting['fast'], EMA_long = setting['slow'], EMA_signal = setting['signal'], dropna=True)
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
        system_output.append({  'setting': setting['name'],
                                'Sharp': trades["Return"].mean() / trades['Return'].std(),
                                'Retrun': trades['StrategyEquity'].iloc[-1],
                                'BuyHold': trades['BuyAndHold'].iloc[-1],
                                'Retrun Std': trades['Return'].std(),
                              })
    syst_out = pd.DataFrame(system_output)
    syst_out.set_index('setting', inplace = True)
    print(syst_out)