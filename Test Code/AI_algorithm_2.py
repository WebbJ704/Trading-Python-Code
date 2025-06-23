import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import ta  
import matplotlib.dates as mdates

def download_data(ticker,api,start_date,end_date):
    filename = f'{ticker}_stock_data.CSV'
    if os.path.exists(filename):
        df = pd.read_csv(filename, parse_dates=['date'], index_col='date').sort_index()
        return df.loc[start_date : end_date].copy()
    else:
        ts = TimeSeries(key=api, output_format="pandas")
        data, _ = ts.get_daily(symbol=ticker, outputsize="full")
        data.columns = ["Open","High","Low","Close","Volume"]
        data = data.reset_index()
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        df = data.sort_index()
        df.to_csv(filename)
        return df.loc[start_date : end_date].copy()

def add_technical_indicators(df):
    df['MA5']  = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['RSI']  = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD']      = macd.macd()
    df['MACD_diff'] = macd.macd_diff()
    df['Volatility']= df['Close'].pct_change().rolling(5).std()
    df.dropna(inplace=True)
    return df

def create_labels(df, forward_days=5, threshold=0.005):
    df['FutOpen'] = df['Open'].shift(-forward_days)
    df['FutRet']  = (df['FutOpen'] - df['Open']) / df['Open']
    df['Signal'] = 'hold'
    df.loc[df['FutRet'] > threshold, 'Signal'] = 'buy'
    df.loc[df['FutRet'] < -threshold, 'Signal'] = 'sell'
    
    # Split data into training and future prediction data
    df_train = df.iloc[:-forward_days].dropna(subset=['FutOpen'])
    df_future = df.iloc[-forward_days:]  # Last `forward_days` rows for prediction
    return df_train, df_future

def preprocess(df):
    features = ['Open','High','Low','Close','Volume','MA5','MA20','RSI','MACD','MACD_diff','Volatility']
    X = df[features].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    le = LabelEncoder().fit(df['Signal'])
    y = le.transform(df['Signal'])
    return Xs, y, le, scaler

def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def backtest(df, y_pred, le):
    df = df.copy()
    df['Pred'] = le.inverse_transform(y_pred)
    df['Position'] = 0
    pos = 0
    for i, sig in enumerate(df['Pred']):
        if sig == 'buy':   pos = 1
        elif sig == 'sell':pos = 0
        df.iat[i, df.columns.get_loc('Position')] = pos
    df['DailyRet'] = df['Open'].pct_change()
    df['StratRet'] = df['DailyRet'] * df['Position'].shift(1).fillna(0)
    df['CumStrat'] = (1 + df['StratRet']).cumprod()
    df['CumHold']  = (1 + df['DailyRet']).cumprod()
    return df

def plot_results(df, ticker):
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8), sharex=True)
    ax1.plot(df.index, df['Open'], label='Open Price', color='tab:blue')

    last_signal = None
    plotted_buy  = False
    plotted_sell = False

    for i in range(1, len(df)):
        sig = df['Pred'].iloc[i]
        price = df['Open'].iloc[i]
        dt    = df.index[i]

        if sig != last_signal:
            if sig == 'buy':
                ax1.scatter(dt, price,
                            marker='^', color='green',
                            s=60, zorder=5,
                            label='Buy' if not plotted_buy else '_nolegend_')
                plotted_buy = True

            elif sig == 'sell':
                ax1.scatter(dt, price,
                            marker='v', color='red',
                            s=60, zorder=5,
                            label='Sell' if not plotted_sell else '_nolegend_')
                plotted_sell = True

            last_signal = sig

    ax1.set_title(f"{ticker} Price & ML Signals")
    ax1.legend()

    ax2.plot(df.index, df['CumStrat'], label='ML Strategy', color='tab:purple')
    ax2.plot(df.index, df['CumHold'],  label='Buy & Hold',  color='tab:orange')
    ax2.set_title("Cumulative Returns")
    ax2.legend()

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, linestyle='--', alpha=0.5)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()



def main():
    ticker     = 'NVDA'
    api_key    = 'SLNEQXVO3S7L9JTH'
    start_date = '2024-05-01'
    end_date   = '2025-04-2'
    forward_days = 5

    df = download_data(ticker, api_key, start_date, end_date)
    df = add_technical_indicators(df)
    df_train, df_future = create_labels(df, forward_days, threshold=0.005)

    # Preprocess training set
    X, y, le, scaler = preprocess(df_train)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = build_model(input_dim=X.shape[1])
    model.fit(
        Xtr, ytr,
        validation_data=(Xte, yte),
        epochs=50, batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=1
    )

    # 1) Predict on training+test for backtest
    y_pred_all = model.predict(X, verbose=0).argmax(axis=1)
    result_backtest = backtest(df_train, y_pred_all, le)

    # 2) Future‐day inference for the last `forward_days`
    feats = ['Open','High','Low','Close','Volume','MA5','MA20','RSI','MACD','MACD_diff','Volatility']
    X_future = scaler.transform(df_future[feats].values)
    y_pred_fut = model.predict(X_future, verbose=0).argmax(axis=1)
    df_future = df_future.copy()
    df_future['Pred'] = le.inverse_transform(y_pred_fut)

    # Append those days to the backtest DataFrame
    full_result = pd.concat([result_backtest, df_future], axis=0)

    plot_results(full_result, ticker)

if __name__=='__main__':
    main()
