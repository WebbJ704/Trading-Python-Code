import Modular_syst as MS
import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from itertools import product


# ----------------- CONFIG -----------------
SYMBOL =  ['AMD']
# ------------------------------------------
if __name__ == "__main__":

 # ----------------- Data Acquisition & Preprocessing -----------------
    df_whole = {}
    train_df = {}
    test_df = {}

    for ticker in SYMBOL:
        # Load raw data once
        df_whole[ticker] = MS.fetch_yf_data(ticker, '2020-01-18')
        # Split data chronologically
        train_df[ticker] =df_whole[ticker][df_whole[ticker].index < '2024-01-01'].copy()
        test_df[ticker] = df_whole[ticker][df_whole[ticker].index >= '2024-01-01'].copy()
 # ----------------- Performance of different variations using optuna ----------------- 
    best_params = {}
    for tickers in SYMBOL:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: MS.objective(trial, train_df[ticker]), n_trials=1000, show_progress_bar=True)
        print(f"Best Sharpe Ratio for {ticker}:", study.best_value)
        print(f"Best Params for {ticker}:", study.best_params)
        best_params[ticker] = study.best_params.values()
  
    trades_optuna , signals_optuna = MS.strat_pipeline(test_df, SYMBOL, best_params, use_MACD = True, use_BB = True)
    for ticker in SYMBOL:
        return_mean = trades_optuna[ticker]['Return'].mean()
        return_std =  trades_optuna[ticker]['Return'].std()
        total_retrun = trades_optuna[ticker]['StrategyEquity'].iloc[-1] 
        sharpe = return_mean/return_std
        print(f'final Strat equity for {ticker} on test data: {total_retrun:.3f}')
        print(f'final Strat sharpe for {ticker} on test data: {sharpe:.3f}')
 # ----------------- Performance of different variations using scipy ----------------- 
    trade_means = MS.variation_perfromace_plot(train_df,SYMBOL,use_MACD = True, use_BB = True)
    params = MS.scipy_optimise(train_df,SYMBOL, trade_means, use_MACD = True, use_BB = True)
    trades_df , signals  =  MS.strat_pipeline(test_df, SYMBOL, params, use_MACD = True, use_BB = True)
    for ticker in SYMBOL:
        return_mean = trades_df[ticker]['Return'].mean()
        return_std =  trades_df[ticker]['Return'].std()
        total_retrun = trades_df[ticker]['StrategyEquity'].iloc[-1] 
        sharpe = return_mean/return_std
        print(f'final Strat equity for {ticker} on test data: {total_retrun:.3f}')
        print(f'final Strat sharpe for {ticker} on test data: {sharpe:.3f}')
  # ----------------- Performance of different variations using scipy ----------------- 



    








