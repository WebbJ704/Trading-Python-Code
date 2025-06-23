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

   
    filename= f'{ticker}_stock_data.CSV'

    # Check if file exists
    if os.path.exists(filename):
        # Read & parse dates
        df = pd.read_csv(filename,parse_dates=['date'],index_col='date').sort_index()
        # True datetime slicing
        return df.loc[start_date : end_date].copy()
    else :
        print(f"Downloading data for {ticker}...")
        ts = TimeSeries(key=api, output_format="pandas")
        data, meta_data = ts.get_daily(symbol=ticker, outputsize="full")
        data.columns = ["Open", "High", "Low", "Close", "Volume"] # Rename columns for easier use
        data = data.reset_index()   # Reset index and format date column

        data["date"] = pd.to_datetime(data["date"])
        data.set_index('date', inplace=True)
        df = data.sort_values("date")  # Sort data in ascending order
        df.to_csv(f'{ticker}_stock_data.CSV') # saving to a csv file 
        return df

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

def create_labels(df, forward_days=5, threshold=0.0):
    # forward return: (Open_t+forward - Open_t)/Open_t
    df['FutOpen'] = df['Open'].shift(-forward_days)
    df['FutRet']  = (df['FutOpen'] - df['Open'])/df['Open']
    df['Signal'] = 'hold'
    df.loc[df['FutRet'] > threshold, 'Signal'] = 'buy'
    df.loc[df['FutRet'] < -threshold, 'Signal'] = 'sell'
    df = df.dropna()
    return df

def preprocess(df):
    features = ['Open','High','Low','Close','Volume','MA5','MA20','RSI','MACD','MACD_diff','Volatility']
    X = df[features].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    le = LabelEncoder().fit(df['Signal'])
    y = le.transform(df['Signal'])
    return Xs, y, le

def build_model(input_dim):
    m = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')  # buy/hold/sell
    ])
    m.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

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
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    print(df.iloc[:,-5:])

    # --- Price & Signals ---
    ax1.plot(df.index, df['Open'], label='Open Price', color='tab:blue')
    buys  = df[df['Pred']=='buy']
    sells = df[df['Pred']=='sell']
    ax1.scatter(buys.index,  buys['Open'],  marker='^', color='green', label='Buy', s=60, zorder=5)
    ax1.scatter(sells.index, sells['Open'], marker='v', color='red',   label='Sell', s=60, zorder=5)
    ax1.set_title(f"{ticker} Price & ML Signals")
    ax1.set_ylabel("Price")
    ax1.legend(loc='upper left')

    # --- Cumulative Returns ---
    ax2.plot(df.index, df['CumStrat'], label='ML Strategy', color='tab:purple')
    ax2.plot(df.index, df['CumHold'],  label='Buy & Hold', color='tab:orange')
    ax2.set_title("Cumulative Returns")
    ax2.set_ylabel("Cumulative Return")
    ax2.legend(loc='upper left')

    # --- Date formatting with AutoLocator & ConciseFormatter ---
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Rotate & layout
    fig.autofmt_xdate()  # auto-rotates labels & right-aligns
    plt.tight_layout()
    plt.show()

def main():
    ticker     = 'AMD'
    api_key    = 'SLNEQXVO3S7L9JTH'
    start_date = '2022-01-01'
    end_date   = '2025-04-11'

    df = download_data(ticker, api_key, start_date, end_date)
    df = add_technical_indicators(df)
    df = create_labels(df, forward_days=5, threshold=0.005)  # 0.5% forward move

    X, y, le = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=False)

    model = build_model(input_dim=X.shape[1])
    es    = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test,y_test),
              epochs=50, batch_size=32, callbacks=[es], verbose=1)

    y_pred = model.predict(X, verbose=0).argmax(axis=1)
    result = backtest(df, y_pred, le)
    plot_results(result, ticker)

if __name__=='__main__':
    main()