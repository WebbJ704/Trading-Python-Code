import data_download as dataDownlaod
import matplotlib.pyplot as plt
import Get_MACD_variations as MACD_var
import Get_RSI_trend_variations as RSI_trend
import Plotting as pl
import Rules as rules
import Signal as Sig
import Get_BB_variations as BB

import backtest as bt

# ----------------- CONFIG -----------------
SYMBOL = 'AAPL'
# ------------------------------------------
if __name__ == "__main__" :
    df = dataDownlaod.fetch_yf_data(SYMBOL,'2024-01-18')
    #plt.plot(df['Close'])
    #plt.show()

    # # mean reversion boiler bangs
    # setting , best_df , best_trades  = BB.BB_variations(df)
    # pl.plot(best_df, best_trades)
    # df = rules.Rules(df, BB_window = setting['window'], BB_std_dev_top = setting['upper'], BB_std_dev_bottom = setting['lower'], dropna=False)
    # df = Sig.generate_signals(df, use_BB=True)
    # trades, signal = bt.backtest(df,use_bb_exit = True)
    # pl.plot(df, trades, BB = True)
    # print(signal[-1])
    # #RSI Trend
    # RSI_settings , best_df , best_trades = RSI_trend.RSI_variations(df)
    # pl.plot(best_df, best_trades)
    # df = rules.Rules(df, rsi_window = RSI_settings['window'], dropna=False)
    # df = Sig.generate_signals(df, use_RSI_trend=True, rsi_trend = RSI_settings['high'],)
    # trades, signal = bt.backtest(df,rsi_trend_exit_threshold = RSI_settings['low'],use_rsi_trend_exit=True)
    # pl.plot(df, trades, RSI =True)
    # print(signal[-1])
    #MACD 
    setting , best_df , best_trades  = MACD_var.MACD_variations(df)
    pl.plot(best_df, best_trades)
    df = rules.Rules(df, EMA_short = setting['fast'], EMA_long = setting['slow'], EMA_signal = setting['signal'], dropna=False)
    df = Sig.generate_signals(df,use_MACD=True)
    trades, signal = bt.backtest(df, use_ema_exit =True)
    pl.plot(df, trades, MACD =True)
    print(signal[-1])
    
   



   

    
    

    

    

  



    