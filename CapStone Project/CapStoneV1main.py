
import CapStoneV2 as CS
import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# ----------------- CONFIG -----------------
SYMBOL = 'AAPL'
# ------------------------------------------
if __name__ == "__main__":
    # Load raw data once
    df = CS.fetch_yf_data(SYMBOL, '2020-01-18', '2025-07-16')

    # Split data chronologically
    train_df = df[df.index < '2024-01-01'].copy()
    test_df = df[df.index >= '2024-01-01'].copy()

    # Create study, pass train and test data to objective using a lambda
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: CS.objective(trial, train_df, test_df), n_trials=100, show_progress_bar=True)

    # Print best results
    print("Best Parameters:", study.best_params)
    print("Best Test Sharpe:", study.best_value)

    # Optional: Show all trials with train/test scores
    results_df = study.trials_dataframe(attrs=("params", "value", "user_attrs"))
    print(results_df[['value', 'user_attrs_train_sharpe', 'user_attrs_test_sharpe']].sort_values(by='value', ascending=False))

    # Run the optimised straegy on the full data set
    df_initial = CS.stochiastic_oscilaotr(df.copy(), k_period=study.best_params['k_period'], d_period=study.best_params['d_period'])
    df_initial = CS.signal_genorator(df_initial, lower_k_threshold=study.best_params['lower_k_threshold'])
    trades_initial, sharp= CS.backtest(df_initial, stochastic_upper=study.best_params['stochastic_upper'])
    bootstarp_results = CS.bootstrap(trades_initial)
    t_stat, p_value, strat_mean_sharp = CS.T_test(trades_initial, df_initial)
    CS.plot_bootstrap(bootstarp_results)
    CS.cumulative_return(trades_initial)

    # Run study, passing objective from CS
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: CS.obj(trial, train_df), n_trials=50)

    # After optimization, access param_results:
    results_df = pd.DataFrame(CS.param_results)
    X = results_df[['k_period', 'd_period', 'lower_k_threshold', 'stochastic_upper']]
    y = results_df['sharpe']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    train_score = rf.score(X_train, y_train)
    print(f"Random Forest R^2 on test set: {train_score:.4f}")

    test_score = rf.score(X_test, y_test)
    print(f"Random Forest R^2 on test set: {test_score:.4f}")

    
    # Now you can predict Sharpe for new parameters, e.g.:
    example_params = pd.DataFrame([{
        'k_period': 14,
        'd_period': 3,
        'lower_k_threshold': 20,
        'stochastic_upper': 80
    }])
    predicted_sharpe = rf.predict(example_params)[0]
    print(f"Predicted Sharpe for example params: {predicted_sharpe:.4f}")






