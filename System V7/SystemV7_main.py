import SystemV7 as V7
import optuna
import numpy as np
import pandas as pd
import warnings
import pandas as pd
from optuna.visualization import plot_optimization_history

# Suppress FutureWarnings from pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

# Suppress info logging from Optuna
optuna.logging.set_verbosity(optuna.logging.CRITICAL)

# ----------------- CONFIG -----------------

no_trials = 800

SYMBOL = [
    # Tech giants ("Magnificent Seven") leading AI/momentum trends
    # 'AAPL',  # Apple
    # 'MSFT',  # Microsoft
    'NVDA',  # NVIDIA
    # 'GOOGL', # Alphabet
    # 'META',  # Meta Platforms
    # 'AMZN',  # Amazon

    # # High-momentum growth/software stocks
    # 'PLTR',  # Palantir – top-performing AI/data analytics
    # 'CRWD',  # CrowdStrike – cybersecurity momentum
    # 'SNOW',  # Snowflake – cloud data platform
    # 'ZS',    # Zscaler – strong upgrades in cybersecurity

    # Semiconductor players integral to AI/trending sectors
    'AMD',   # Advanced Micro Devices – AI-driven rebound
    #'AVGO',  # Broadcom – chips & tech infrastructure
]

# ------------------------------------------
if __name__ == "__main__":

 # ----------------- Data Acquisition & Preprocessing -----------------
    df_whole = {}
    train_df = {}
    test_df = {}

    for ticker in SYMBOL:
        # Load raw data once
        df_whole[ticker] = V7.fetch_yf_data(ticker, '2020-01-18')
        # Split data chronologically
        train_df[ticker] =df_whole[ticker][df_whole[ticker].index < '2024-01-01'].copy()
        test_df[ticker] = df_whole[ticker][df_whole[ticker].index >= '2024-01-01'].copy()

 #----------------- Performance of different variations -----------------  
       
    #spread of weights, risk , retrun which can be used ti determin min retrun constrain for Scipy minimise
   # _ = V7.variation_perfromace_plot(df_whole, SYMBOL, n_trials=100)

 # ----------------- Performance of different variations using optuna ----------------- 
    best_params = {}
    for ticker in SYMBOL:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: V7.objective(trial, train_df[ticker]), n_trials=no_trials, show_progress_bar=True)
        # fig = plot_optimization_history(study)
        # fig.show()
        print(f"Best anual Sharpe Ratio for {ticker} on train data:", study.best_value)
        print(f"Best Params for {ticker} on train data:", study.best_params)
        best_params[ticker] = study.best_params.values()

    trades_optuna , signals_optuna, df_optuna_train = V7.strat_pipeline(train_df, SYMBOL, best_params)
    porfolio_df_train = pd.DataFrame({ticker: df_optuna_train[ticker]['MarketReturn'] for ticker in SYMBOL})
  
    for ticker in SYMBOL:
        df_optuna_train[ticker].index = pd.to_datetime(df_optuna_train[ticker].index)
        daily_returns = df_optuna_train[ticker]["StrategyReturn"].dropna()
        total_retrun = df_optuna_train[ticker]['Equity'].iloc[-1] 
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)  
        roll_max = df_optuna_train[ticker]["Equity"].cummax()
        drawdown = df_optuna_train[ticker]["Equity"]/roll_max - 1
        max_dd = drawdown.min()*100
        print(f'final Strat equity for {ticker} on train data: {total_retrun:.3f}')
        print(f'final Strat anual sharpe for {ticker} on train data: {sharpe:.3f}')
        print(f'Max Donwndraw for {ticker} on train data:{max_dd:.1f}% want < 15% , good is  < 20-25%')
    
    trades_optuna , signals_optuna, df_optuna_test = V7.strat_pipeline(test_df, SYMBOL, best_params)
    print(signals_optuna)
    porfolio_df_test = pd.DataFrame({ticker: df_optuna_test[ticker]['MarketReturn'] for ticker in SYMBOL})
    for ticker in SYMBOL:
        df_optuna_test[ticker].index = pd.to_datetime(df_optuna_test[ticker].index)
        daily_returns = df_optuna_test[ticker]["StrategyReturn"].dropna()
        total_retrun = df_optuna_test[ticker]['Equity'].iloc[-1] 
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)  
        roll_max = df_optuna_test[ticker]["Equity"].cummax()
        drawdown = df_optuna_test[ticker]["Equity"]/roll_max - 1
        max_dd = drawdown.min()*100
        print(f'final Strat equity for {ticker} on test data: {total_retrun:.3f}')
        print(f'final Strat anual sharpe for {ticker} on test data: {sharpe:.3f}')
        print(f'Max Donwndraw for {ticker} on test data:{max_dd:.1f}% want < 15% , good is  < 20-25%')

   # ----------------- Portfolio Performance Evaluation and optimisation with Optuna -----------------    

    #creates a study of different weights to find best using beysion theroy on train data
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: V7.objective_weights(trial, porfolio_df_train, SYMBOL), n_trials=no_trials, show_progress_bar=True)
    fig = plot_optimization_history(study)
    fig.show()
    weights_optuna = study.best_params.values()
    weights = np.array(list(weights_optuna))
    weights = weights/sum(weights)
    df_weights = pd.DataFrame({ "Ticker": SYMBOL, "Weight": weights })
    print("Best Sharpe Ratio:", study.best_value)
    print("Best weights:")
    print(df_weights)
   
    #run optimal weights on test data and whole dataset
    anual_sharpe_train, Porfolio_Returns_Train = V7.portfolio_performance_pipeline(porfolio_df_train,weights_optuna, SYMBOL)
    anual_sharpe_test, Porfolio_Returns_test = V7.portfolio_performance_pipeline(porfolio_df_test,weights_optuna, SYMBOL)
    print(f'Anualsied sharpe on Train data of optimised weights from study', anual_sharpe_train)
    print(f'Anualsied sharpe on Test data of optimised weights from study', anual_sharpe_test)
    print(f'Return on Train data of optimised weights from study', Porfolio_Returns_Train.sum()*100,'%')
    print(f'Return on Test data of optimised weights from study', Porfolio_Returns_test.sum()*100, '%')

    # run bootsrap on retruns on whole df and test df using optuna weights
    Bootstrap_results_train_optuna =  V7.bootstrap_weighted(Porfolio_Returns_Train)
    Bootstrap_results_test_optuna =  V7.bootstrap_weighted(Porfolio_Returns_test)
    V7.plot_bootstrap_weights(Bootstrap_results_test_optuna, SYMBOL, weights_optuna, 'Test Data returns: Optuna Weights')
    V7.plot_bootstrap_weights(Bootstrap_results_train_optuna, SYMBOL, weights_optuna, 'Train Data returns: Optuna Weights')

    #plot the cumulative retruns of each stock vs portfolio weighted return
    V7.cumulative_return_plot(porfolio_df_train,Porfolio_Returns_Train, SYMBOL, weights_optuna, 'Cumulative Return: optimal weights from optuna Study on Train Dataset')
    V7.cumulative_return_plot(porfolio_df_test,Porfolio_Returns_test, SYMBOL, weights_optuna, 'Cumulative Return: optimal weights from optuna Study on test Dataset')