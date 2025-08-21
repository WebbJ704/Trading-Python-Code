
import CapStoneV2 as CS
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
SYMBOL = 'SPY'
# ------------------------------------------
if __name__ == "__main__":
    # Load raw data once
    df = CS.fetch_yf_data(SYMBOL, '2020-01-18', '2025-07-16')

    # Split data chronologically
    train_df = df[df.index < '2024-01-01'].copy()
    test_df = df[df.index >= '2024-01-01'].copy()

    # Create study, pass train and test data to objective using a lambda
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: CS.objective(trial, train_df, test_df), n_trials=50, show_progress_bar=True)

    # Print best results
    print("Best Parameters:", study.best_params)
    print("Best Test Sharpe:", study.best_value)

    # Run the optimised straegy on the full data set
    df_initial = CS.stochiastic_oscilaotr(df.copy(), k_period=study.best_params['k_period'], d_period=study.best_params['d_period'])
    df_initial = CS.signal_genorator(df_initial, lower_k_threshold=study.best_params['lower_k_threshold'])
    trades_initial, sharp= CS.backtest(df_initial, stochastic_upper=study.best_params['stochastic_upper'])
    bootstarp_results = CS.bootstrap(trades_initial)
    t_stat, p_value, strat_mean_sharp = CS.T_test(trades_initial, df_initial)
    CS.plot_bootstrap(bootstarp_results)
    CS.cumulative_return(trades_initial)

    # After optimization, access param_results for Model:
    results_df = study.trials_dataframe(attrs=("params", "value"))
    X = results_df.drop(columns=["value"])
    y = results_df["value"] # can we try and optimise for both sharp and retruns 
    # Drop bad trials
    X = X[y.notna()]
    y = y[y.notna()]
    
    # split the data for model into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # initlaise and fit the model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Model scores 
    train_score = rf.score(X_train, y_train)
    print(f"Random Forest R^2 on train set: {train_score:.4f}")
    test_score = rf.score(X_test, y_test)
    print(f"Random Forest R^2 on test set: {test_score:.4f}")

    # Predict over grid
    new_params = pd.DataFrame(product(range(5, 30), range(2, 10), range(10,30), range(70,90) ), columns=['params_d_period', 'params_k_period', 'params_lower_k_threshold','params_stochastic_upper'])
    new_params['predicted_sharpe'] = rf.predict(new_params)

    # find best 20 paramaters for best sharpe
    best = new_params.sort_values('predicted_sharpe', ascending=False).head(20)

    # use a radom sample from best params to see how perfroms on test_df
    sample = best.sample(n=1)
    sample_d_period = int(sample.iloc[0]['params_d_period'])
    sample_k_period = int(sample.iloc[0]['params_k_period'])
    sample_lower_k_threshhold = int(sample.iloc[0]['params_lower_k_threshold'])
    sample_stochastic_upper = int(sample.iloc[0]['params_stochastic_upper'])

    # Run the sampled straegy from random forset on the test data set
    df_sample_test = CS.stochiastic_oscilaotr(test_df, k_period=sample_k_period, d_period=sample_d_period)
    df_sample_test= CS.signal_genorator(df_sample_test, lower_k_threshold=sample_lower_k_threshhold)
    trades_sample_test, sharp= CS.backtest(df_sample_test, stochastic_upper=sample_stochastic_upper)
    bootstarp_results_sample = CS.bootstrap(trades_sample_test)
    t_stat, p_value, strat_mean_sharp = CS.T_test(trades_sample_test, df_sample_test)
    CS.plot_bootstrap(bootstarp_results_sample)
    CS.cumulative_return(trades_sample_test)

    """use varaicne to understand how the change in paramters effects the sharp ratio"""
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit regression on full feature set
    reg1 = LinearRegression()
    reg1.fit(X_scaled, y)
    y_pred_all = reg1.predict(X_scaled)
    all_features_r2 = r2_score(y, y_pred_all)
    print(f'Linear Regression R^2 on full feature set: {all_features_r2:.4f}')

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=4, random_state=22)
    X_pca = pca.fit_transform(X_scaled)

    # Fit regression on PCA-transformed features
    reg2 = LinearRegression()
    reg2.fit(X_pca, y)
    y_pred_pca = reg2.predict(X_pca)
    pca_r2 = r2_score(y, y_pred_pca)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f'Linear Regression R^2 on PCA-transformed features: {pca_r2:.4f}')
    print(f'The first four components explain {explained_var * 100:.2f}% of the variance in the features.')
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance by PCA Components")
    plt.grid(True)
    plt.show()







