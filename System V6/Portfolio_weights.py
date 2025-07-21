import pandas as pd
import numpy as np
from scipy.optimize import minimize

def optimize_portfolio_weights_from_dataframe(trades: pd.DataFrame, 
                                              user_weights: dict = None,
                                              risk_free_rate: float = 0.00):
    """
    Optimize portfolio weights using mean-variance optimization on trade-level strategy returns.
    Or, compute performance using user-defined weights.

    Parameters:
        trades (pd.DataFrame): Must contain ['SYMBOL', 'ExitDate', 'Return'] columns.
        user_weights (dict, optional): Dictionary of weights keyed by 'SYMBOL'. If provided, skips optimization.
        risk_free_rate (float): Annual risk-free rate (default: 0.00)

    Returns:
        weights (pd.Series): Optimal or user-defined weights per symbol.
        sharpe (float): Annualized Sharpe ratio.
        equity_curve (pd.Series): Portfolio cumulative equity curve.
    """
    # Ensure ExitDate is datetime
    trades['ExitDate'] = pd.to_datetime(trades['ExitDate'])

    # Pivot: each symbol's return indexed by ExitDate
    pivot = trades.pivot_table(index='ExitDate', columns='SYMBOL', values='Return', aggfunc='sum')
    pivot = pivot.sort_index().fillna(0)

    mean_returns = pivot.mean() * 252 / pivot.shape[0]
    cov_matrix = pivot.cov() * 252 / pivot.shape[0]
    num_assets = len(mean_returns)

    if user_weights is not None:
        # Use provided weights
        weights = pd.Series(user_weights)
        weights = weights.reindex(pivot.columns).fillna(0)
        weights = weights / weights.sum()  # Normalize just in case
    else:
        # Optimize weights
        def portfolio_performance(weights):
            ret = np.dot(weights, mean_returns)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return ret, vol

        def neg_sharpe_ratio(weights):
            ret, vol = portfolio_performance(weights)
            return -(ret - risk_free_rate) / vol

        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        init_guess = [1.0 / num_assets] * num_assets

        result = minimize(neg_sharpe_ratio, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = pd.Series(result.x, index=pivot.columns)

    # Compute equity curve using weights
    portfolio_returns = pivot @ weights
    equity_curve = (1 + portfolio_returns).cumprod()

    ann_return = portfolio_returns.mean() * 252
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol

    return weights.round(4), round(sharpe, 4), equity_curve
