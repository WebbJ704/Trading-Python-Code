"""
# Quantitative Trading Strategy Optimization with Machine Learning

This project builds a robust **multi-asset trading framework** that combines **technical indicator rule optimization**, **machine learning-based hyperparameter tuning**, and **portfolio construction** via **mean-variance optimization**.

It’s designed to develop and evaluate rule-based strategies for different stocks, generate signals, backtest results, and optimize portfolio allocation.

---

## Key Features

- **Technical Indicator Optimization**  
  - MACD, Bollinger Bands, and more  
  - Evaluates variations via backtesting

- **Machine Learning for Hyperparameter Tuning**  
  - Uses optimization techniques (e.g., Bayesian optimization, grid/random search) to find best indicator parameters  
  - Integrates easily with `scikit-learn` or custom models

- **Custom Signal Generation**  
  - Rule-based logic for MACD, Bollinger Bands, etc.  
  - Easily extendable with RSI, ADX, Stochastic, etc.

- **Backtesting Engine**  
  - Entry/exit logic, trade evaluation  
  - Bootstrapped Sharpe ratio + equity curve visualization

- **Portfolio Optimization**  
  - Mean-variance optimization (Sharpe-maximizing weights)  
  - Option to input manual weights

---

## Machine Learning Component

- **Hyperparameter Optimization** is used to tune strategy parameters (e.g., MACD fast/slow periods) for each stock.
- Techniques used:
  - Random Search / Grid Search
  - Bayesian Optimization (e.g., using `Optuna` or `scikit-optimize`)
- Evaluation is based on maximizing risk-adjusted return (e.g., Sharpe ratio) over bootstrapped samples.

---

## Project Structure

```bash
.
├── MACD_var.py           # MACD + ML-based parameter optimization
├── rules.py              # Trading logic rules (indicator-based)
├── signals.py            # Signal generator (MACD, BB, etc.)
├── backtest.py           # Backtesting engine
├── bootstrap.py          # Bootstrap sampling
├── plotting.py           # Plotting utilities
├── data_Download.py      # Yahoo Finance downloader
├── Portfolio_weights.py  # Portfolio weight optimization
├── main.py               # End-to-end strategy runner
├── Trades/               # Output directory for trades per stock
├── README.md             # This file
