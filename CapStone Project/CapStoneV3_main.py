import CapStoneV3 as CS
import pandas as pd
import optuna
import numpy as np
import warnings
from optuna.visualization import plot_optimization_history

# Suppress FutureWarnings from pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

# Suppress info logging from Optuna
optuna.logging.set_verbosity(optuna.logging.CRITICAL)

# ----------------- INNSTRUMENT CONFIG -----------------
SYMBOL = [
    # Tech giants ("Magnificent Seven") leading AI/momentum trends
    'AAPL',  # Apple
    'MSFT',  # Microsoft
    'NVDA',  # NVIDIA
    'GOOGL', # Alphabet
    'META',  # Meta Platforms
    'AMZN',  # Amazon

    # High-momentum growth/software stocks
    'PLTR',  # Palantir – top-performing AI/data analytics
    'CRWD',  # CrowdStrike – cybersecurity momentum
    'SNOW',  # Snowflake – cloud data platform
    'ZS',    # Zscaler – strong upgrades in cybersecurity

    # Semiconductor players integral to AI/trending sectors
    'AMD',   # Advanced Micro Devices – AI-driven rebound
    'AVGO',  # Broadcom – chips & tech infrastructure
]
# ------------------------------------------
if __name__ == "__main__":
    # ----------------- Data Acquisition & Preprocessing -----------------
    df = {}
    df_return = {}
    for ticker in SYMBOL:
        data = CS.fetch_yf_data(ticker, '2020-01-18') #This is the start date of where you want data to be at
        df[f'{ticker} Adj Close'] = data['Close']
        df[f'{ticker} Return'] = data['Close'].pct_change()
        df_return[f'{ticker}'] = data['Close'].pct_change()
    df =pd.DataFrame(df)
    df.dropna(inplace =True) 
    df_return = pd.DataFrame(df_return)
    df_return.dropna(inplace = True)

    # Split data chronologically
    train_df = df[df.index < '2024-01-01'].copy()
    test_df = df[df.index >= '2024-01-01'].copy() 
    test_return_df = df_return[df_return.index < '2024-01-01'].copy()
    train_retrun_df = df_return[df_return.index >= '2024-01-01'].copy()

    # ----------------- Single stock Performance Evaluation / Statistical Analysis -----------------

    # Bootstrap Whole data to better understand each stock for whole dataset
    # Bootstrap_results =  CS.bootstrap(df, SYMBOL)
    # CS.plot_bootstrap(Bootstrap_results, SYMBOL)

    #Determin Anualised shapre for each stock on whole dataset
    Anualised_sharp = CS.Anualised_sharpe(df,SYMBOL)
    print(Anualised_sharp)
 
    #----------------- Portfolio Performance of different weights -----------------  
       
    #spread of weights, risk , retrun which can be used ti determin min retrun constrain for Scipy minimise
    CS.weights_perfromace_plot(df_return, SYMBOL)
   
    # ----------------- portfolio Performance Evaluation with scipy minimise with a min return constrain -----------------    

    # scipiy optimise on test data then input weights into train data
    sharpe_scipy , weights_scipy = CS.optimise(test_return_df, SYMBOL, risk_free_rate=0.02, min_return=0.25)
    weights = np.array(list(weights_scipy))
    weights = weights/sum(weights)
    weights_str_scipy = ", ".join([f"{t}: {w:.2f}" for t, w in zip(SYMBOL, weights)])
    print('Sharpe from Scipy',sharpe_scipy)
    print('weight from scipy',weights_str_scipy)

    anual_sharpe_train_scipy, Porfolio_Returns_train_scipy = CS.portfolio_performance_pipeline(train_df,weights_scipy, SYMBOL)
    anual_sharpe_test_scipy, Porfolio_Returns_test_scipy = CS.portfolio_performance_pipeline(test_df,weights_scipy, SYMBOL)
    print(f'Anualsied sharpe on train data of optimised weights from scipy', anual_sharpe_train_scipy)
    print(f'Anualsied sharpe on Test data of optimised weights from scipy', anual_sharpe_test_scipy)
    print(f'Return on train data of optimised weights from scipy', Porfolio_Returns_train_scipy.sum())
    print(f'Return on Test data of optimised weights from scipy', Porfolio_Returns_test_scipy.sum())

    # # run bootsrap on retruns on train returns and test returns using scipy weights
    # Bootstrap_results_train_scipy =  CS.bootstrap_weighted(Porfolio_Returns_train_scipy)
    # Bootstrap_results_test_scipy =  CS.bootstrap_weighted(Porfolio_Returns_test_scipy)
    # CS.plot_bootstrap_weights(Bootstrap_results_train_scipy, SYMBOL, weights_scipy, 'Train Data returns: scipy Weights')
    # CS.plot_bootstrap_weights(Bootstrap_results_test_scipy, SYMBOL, weights_scipy, 'Test Data returns: scipy Weights')
   
    #plot the cumulative retruns of each stock vs portfolio weighted return
    CS.cumulative_return_plot(train_df,Porfolio_Returns_train_scipy, SYMBOL, weights_scipy, 'Cumulative Return: optimal weights from scipy Study on train Dataset')
    CS.cumulative_return_plot(test_df,Porfolio_Returns_test_scipy, SYMBOL, weights_scipy, 'Cumulative Return: optimal weights from scipy Study on test Dataset')


    # ----------------- portfolio Performance Evaluation and optimisation with Optuna -----------------    

    #creates a study of different weights to find best using beysion theroy on train data
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: CS.objective(trial, train_df, SYMBOL), n_trials=1000, show_progress_bar=True)
    fig = plot_optimization_history(study)
    fig.show()
    weights_optuna = study.best_params.values()
    weights = np.array(list(weights_optuna))
    weights = weights/sum(weights)
    weights_str = ", ".join([f"{t}: {w:.2f}" for t, w in zip(SYMBOL, weights)])
    print("Best Sharpe Ratio:", study.best_value)
    print("Best weights:", weights)

    #run optimal weights on test data and whole dataset
    anual_sharpe_whole, Porfolio_Returns_whole = CS.portfolio_performance_pipeline(df,weights_optuna, SYMBOL)
    anual_sharpe_test, Porfolio_Returns_test = CS.portfolio_performance_pipeline(test_df,weights_optuna, SYMBOL)
    print(f'Anualsied sharpe on whole data of optimised weights from study', anual_sharpe_whole)
    print(f'Anualsied sharpe on Test data of optimised weights from study', anual_sharpe_test)
    print(f'Return on whole data of optimised weights from study', Porfolio_Returns_whole.sum())
    print(f'Return on Test data of optimised weights from study', Porfolio_Returns_test.sum())
    
    # # run the bootstrap on the optimal weights and returns from optuna study
    # Bootstrap_results =  CS.bootstrap_weighted(study.best_trial.user_attrs.get("portfolio_returns"))
    # CS.plot_bootstrap_weights(Bootstrap_results, SYMBOL, weights_optuna, 'optuna returns and weights')

    # # run bootsrap on retruns on whole df and test df using optuna weights
    # Bootstrap_results_test_optuna =  CS.bootstrap_weighted(Porfolio_Returns_whole)
    # Bootstrap_results_whole_optuna =  CS.bootstrap_weighted(Porfolio_Returns_test)
    # CS.plot_bootstrap_weights(Bootstrap_results_test_optuna, SYMBOL, weights_optuna, 'Test Data returns: Optuna Weights')
    # CS.plot_bootstrap_weights(Bootstrap_results_whole_optuna, SYMBOL, weights_optuna, 'Whole Data returns: Optuna Weights')

    #plot the cumulative retruns of each stock vs portfolio weighted return
    CS.cumulative_return_plot(df,Porfolio_Returns_whole, SYMBOL, weights_optuna, 'Cumulative Return: optimal weights from optuna Study on Whole Dataset')
    CS.cumulative_return_plot(test_df,Porfolio_Returns_test, SYMBOL, weights_optuna, 'Cumulative Return: optimal weights from optuna Study on test Dataset')

    # ----------------- portfolio Performance Evaluation with RandomForest Modeling -----------------    

    #After study acces params to train a random forest model to predict new weightings
    res_df = study.trials_dataframe(attrs = ('params','value'))

    #Use Random forest modeling to predict best weights to maximise sharpe ratio
    weights_model = CS.RandomForest(res_df,SYMBOL)
    weights = np.array(list(weights_model))
    weights = weights/sum(weights)
    weights_str = ", ".join([f"{t}: {w:.4f}" for t, w in zip(SYMBOL, weights)])
    print("Model weights Randomly selected from Top 20 to avoid overfitting:", weights_str)
   
    #run models through evaluatating pipile on test df and whole df
    anual_sharpe_whole_model, Porfolio_Returns_whole_model = CS.portfolio_performance_pipeline(df,weights_model, SYMBOL)
    anual_sharpe_test_model, Porfolio_Returns_test_model = CS.portfolio_performance_pipeline(test_df,weights_model, SYMBOL)
    print(f'Anualsied sharpe on whole data of sampled weights from model', anual_sharpe_whole_model)
    print(f'Anualsied sharpe on Test data of sampled weights from model', anual_sharpe_test_model)
    print(f'Return on whole data of sampled weight from model {Porfolio_Returns_whole.sum()*100:.2f} %')
    print(f'Return on Test data of sampled weights from model {Porfolio_Returns_test.sum()*100:.2f} %')

    # # run the bootstrap on the optimal weights return from random forest on test data and whole data
    # Bootstrap_results_test_model =  CS.bootstrap_weighted(Porfolio_Returns_test_model)
    # Bootstrap_results_whole_model =  CS.bootstrap_weighted(Porfolio_Returns_test_model)
    # CS.plot_bootstrap_weights(Bootstrap_results_test_model, SYMBOL, weights_model,'Test Data returns: Model Weights')
    # CS.plot_bootstrap_weights(Bootstrap_results_whole_model, SYMBOL, weights_model,'Whole Data returns: Model Weights')

    #plot the cumulative retruns of each stock vs portfolio weighted return
    CS.cumulative_return_plot(df,Porfolio_Returns_whole_model, SYMBOL, weights_model, "Cumulative Return: sampled weights from Random Forest Model on Whole Dataset")
    CS.cumulative_return_plot(test_df,Porfolio_Returns_test_model, SYMBOL, weights_model, 'Cumulative Return: sampled weights from Random Forest Model on test Dataset') 




    

    
    

