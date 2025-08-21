import data_download as dataDownlaod
import matplotlib.pyplot as plt
from Variations import MACD_BB_variations as var
import Portfolio_weights as PW
import Plotting as pl
import Rules as rules
import Signal as Sig
import backtest as bt
import bootstrap as bs
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
SYMBOL = ['AMD','NVDA']
# ------------------------------------------
if __name__ == "__main__" :
    Trades_stocks_train = []
    Trades_stocks_test = []
    for ticker in SYMBOL:
        df_train = dataDownlaod.fetch_yf_data(ticker,'2023-01-18','2024-06-30')
    
        #Finding variations and bakctesting on train data
        setting , best_df , best_trades = var.MACD_variations(df_train)
        df_train = rules.Rules(df_train, EMA_short = setting['fast'], EMA_long = setting['slow'], EMA_signal = setting['signal'], 
                        BB_window = setting['window'], BB_std_dev_top = setting['upper'], BB_std_dev_bottom = setting['lower'], dropna=False)
        df_train = Sig.generate_signals(df_train,
                                    use_ADX=False,
                                    use_RSI=False,
                                    use_EMA=False,
                                    use_MACD=True,
                                    use_ATR=False,
                                    use_volume_spike=False,
                                    use_stochastic=False,
                                    use_BB=True )
        trades, signal = bt.backtest(df_train,use_macd_exit = True, use_bb_exit = True)
        results, simulations = bs.bootstrap(trades)
        trades = trades.reset_index(drop=False) 
        trades['SYMBOL'] = ticker  # Important for optimization
        Trades_stocks_train.append(trades)


        #out of sample testing 
        df_test = dataDownlaod.fetch_yf_data(ticker,'2024-06-30')
        print('Final seetting used on test data',pd.DataFrame([setting]).set_index('name'))
        df_test = rules.Rules(df_test, EMA_short = setting['fast'], EMA_long = setting['slow'], EMA_signal = setting['signal'], 
                                BB_window = setting['window'], BB_std_dev_top = setting['upper'], BB_std_dev_bottom = setting['lower'], dropna=False)
        df_test = Sig.generate_signals(df_test,
                                    use_ADX=False,
                                    use_RSI=False,
                                    use_EMA=False,
                                    use_MACD=True,
                                    use_ATR=False,
                                    use_volume_spike=False,
                                    use_stochastic=False,
                                    use_BB=True )
        trades, signal = bt.backtest(df_test,use_macd_exit = True,use_bb_exit = True)
        results, simulations = bs.bootstrap(trades)
        pl.plot(df_test, trades, results, simulations, ticker, BBMACD =True)
        print(signal[-1])
        trades = trades.reset_index(drop=False) 
        trades['SYMBOL'] = ticker  # Important for optimization
        Trades_stocks_test.append(trades)

    # Find weights from train data
    all_trades = pd.concat(Trades_stocks_train, ignore_index=True)  
    weights, sharpe, equity = PW.optimize_portfolio_weights_from_dataframe(all_trades)
    print("Optimized Weights From Back Test train data:\n", weights)
    print(f"Sharpe Ratio of backtest weights on train darta: {sharpe:.2f}")

    #Perform on out of weights found from backtest
    all_trades = pd.concat(Trades_stocks_test, ignore_index=True)  
    weights, sharpe, equity = PW.optimize_portfolio_weights_from_dataframe(all_trades, user_weights=weights)
    equity.plot(title=f'Strategy Equity Curve on Out-of-Sample Test{weights}', legend=False)
    print("Optimized Weights From Back Teston test data:\n", weights)
    print(f"Sharpe Ratio of weights on out of sample test: {sharpe:.2f}")
    plt.show()

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
    
   



   

    
    

    

    

  



    