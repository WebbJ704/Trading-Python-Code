import data_download as dataDownlaod
import matplotlib.pyplot as plt
import Get_MACD_variations as MACD_var
import Plotting as pl
import Rules as rules
import Signal as Sig
import numpy as np
import backtest as bt

# ----------------- CONFIG -----------------
SYMBOL = 'AAPL'
# ------------------------------------------
if __name__ == "__main__" :
    df = dataDownlaod.fetch_yf_data(SYMBOL,'2020 -09-18')
    plt.plot(df['Adj Close'])
    plt.show()
    setting , best_df , best_trades  = MACD_var.MACD_variations(df)
    pl.plot(best_df, best_trades)
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
    pl.plot(df, trades)
    print(signal[-1])


    