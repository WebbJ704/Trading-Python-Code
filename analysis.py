import pandas as pd 
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
import os

def donwload_data(tickers,start_date,end_date,api):
   for ticker in tickers:
    filename= f'{ticker}_stock_data.CSV'
    # Check if file exists
    if os.path.exists(filename):
        print(f" {filename} already exists. Skipping download.")
        df = pd.read_csv(filename)
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        df.set_index("date", inplace=True)
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


def combine_data(tickers, start_date, end_date):
    ticker_str = "_".join(tickers)
    file = f'{ticker_str}_combined_data.csv'

    if os.path.exists(file):
        print(f" {file} already exists. Skipping combine.")
        df = pd.read_csv(file, index_col='Date', parse_dates=True)
        return df
    else:
        combined_data = pd.DataFrame()

        for ticker in tickers:
            filename = f'{ticker}_stock_data.CSV'
            df = pd.read_csv(filename, index_col='date', parse_dates=True)
            df.index = pd.to_datetime(df.index)
            df.index = df.index.date

            # Keep Close and Volume
            df = df[['Close', 'Volume','High','Low']]
            df.rename(columns={
                'Close': f'{ticker} Close',
                'Volume': f'{ticker} Volume',
                'High': f'{ticker} High',
                'Low': f'{ticker} Low'
            }, inplace=True)

            if combined_data.empty:
                combined_data = df
            else:
                combined_data = combined_data.join(df, how='outer')

        combined_data.index = pd.to_datetime(combined_data.index)
        combined_data.index.name = 'Date'
        combined_data = combined_data.dropna()
        combined_data = combined_data[
            (combined_data.index >= pd.to_datetime(start_date)) &
            (combined_data.index <= pd.to_datetime(end_date))
        ]

        combined_data.to_csv(file)
        print(f"Combined data saved to {file}")
        return combined_data

def analyze_stock(ticker, df):
    decision_scores = {}

    for tick in ticker:
        new_cols = {}  # Store all new columns in a dictionary

        # Compute technical indicators
        new_cols[f'{tick} SMA_20'] = df[f'{tick} Close'].rolling(window=20).mean()
        new_cols[f'{tick} EMA_20'] = df[f'{tick} Close'].ewm(span=20, adjust=False).mean()
        new_cols[f'{tick} Momentum'] = df[f'{tick} Close'] - df[f'{tick} Close'].shift(10)

        # Stochastic Oscillator
        k_period = 14
        d_period = 3
        low_min = df[f'{tick} Low'].rolling(window=k_period).min()
        high_max = df[f'{tick} High'].rolling(window=k_period).max()
        new_cols[f'{tick} %K'] = 100 * ((df[f'{tick} Close'] - low_min) / (high_max - low_min))
        new_cols[f'{tick} %D'] = new_cols[f'{tick} %K'].rolling(window=d_period).mean()

        # OBV
        obv = [0]
        for i in range(1, len(df)):
            if df[f'{tick} Close'].iloc[i] > df[f'{tick} Close'].iloc[i - 1]:
                obv.append(obv[-1] + df[f'{tick} Volume'].iloc[i])
            elif df[f'{tick} Close'].iloc[i] < df[f'{tick} Close'].iloc[i - 1]:
                obv.append(obv[-1] - df[f'{tick} Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        new_cols[f'{tick} OBV'] = pd.Series(obv, index=df.index)

        # ADX
        n = 14
        high = df[f'{tick} High']
        low = df[f'{tick} Low']
        close = df[f'{tick} Close']

        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)

        atr = tr.rolling(n).mean()
        plus_di = 100 * (plus_dm.rolling(n).sum() / atr)
        minus_di = abs(100 * (minus_dm.rolling(n).sum() / atr))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        new_cols[f'{tick} ADX'] = dx.rolling(n).mean()

        # RSI
        delta = df[f'{tick} Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        new_cols[f'{tick} RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df[f'{tick} Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df[f'{tick} Close'].ewm(span=26, adjust=False).mean()
        new_cols[f'{tick} MACD'] = ema_12 - ema_26
        new_cols[f'{tick} Signal'] = new_cols[f'{tick} MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        mid = df[f'{tick} Close'].rolling(20).mean()
        std = df[f'{tick} Close'].rolling(20).std()
        new_cols[f'{tick} BB_Mid'] = mid
        new_cols[f'{tick} BB_Upper'] = mid + 2 * std
        new_cols[f'{tick} BB_Lower'] = mid - 2 * std

        # Volume change
        new_cols[f'{tick} Vol_Change'] = df[f'{tick} Volume'].pct_change()

        # Add all new columns at once to the DataFrame
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        # === Scoring Logic ===
        latest = df.iloc[-1]
        score = {'Buy': 0, 'Hold': 0, 'Sell': 0}

        # OBV trend
        if df[f'{tick} OBV'].iloc[-1] > df[f'{tick} OBV'].iloc[-5]:
            score['Buy'] += 1
        elif df[f'{tick} OBV'].iloc[-1] < df[f'{tick} OBV'].iloc[-5]:
            score['Sell'] += 1
        else:
            score['Hold'] += 1

        # OBV & price confirmation
        price_trend = df[f'{tick} Close'].iloc[-1] > df[f'{tick} Close'].iloc[-5]
        obv_trend = df[f'{tick} OBV'].iloc[-1] > df[f'{tick} OBV'].iloc[-5]
        if price_trend and obv_trend:
            score['Buy'] += 1
        elif not price_trend and not obv_trend:
            score['Sell'] += 1
        else:
            score['Hold'] += 1

        # Stochastic Oscillator
        if latest[f'{tick} %K'] < 20 and latest[f'{tick} %D'] < 20:
            score['Buy'] += 1
        elif latest[f'{tick} %K'] > 80 and latest[f'{tick} %D'] > 80:
            score['Sell'] += 1
        else:
            score['Hold'] += 1

        # ADX
        if latest[f'{tick} ADX'] > 25:
            score['Buy'] += 1
        else:
            score['Hold'] += 1

        # RSI
        if latest[f'{tick} RSI'] < 30:
            score['Buy'] += 1
        elif latest[f'{tick} RSI'] > 70:
            score['Sell'] += 1
        else:
            score['Hold'] += 1

        # MACD
        if latest[f'{tick} MACD'] > latest[f'{tick} Signal']:
            score['Buy'] += 1
        else:
            score['Sell'] += 1

        # Momentum
        if latest[f'{tick} Momentum'] > 0:
            score['Buy'] += 1
        else:
            score['Sell'] += 1

        # Price vs EMA
        if latest[f'{tick} Close'] > latest[f'{tick} EMA_20']:
            score['Buy'] += 1
        else:
            score['Sell'] += 1

        # Bollinger Bands
        if latest[f'{tick} Close'] < latest[f'{tick} BB_Lower']:
            score['Buy'] += 1
        elif latest[f'{tick} Close'] > latest[f'{tick} BB_Upper']:
            score['Sell'] += 1
        else:
            score['Hold'] += 1

        # Normalize
        total = sum(score.values())
        for key in score:
            score[key] = round((score[key] / total) * 100, 2)

        decision_scores[tick] = score

    return df, decision_scores


def visualize_stock(ticker_list, df):
    for ticker in ticker_list:
        plt.figure(figsize=(16, 24))
        
        # Subplot 1: Price with SMA and EMA
        plt.subplot(5, 1, 1)
        plt.plot(df.index, df[f'{ticker} Close'], label='Close Price', color='black')
        plt.plot(df.index, df[f'{ticker} SMA_20'], label='SMA 20', color='blue', linestyle='--')
        plt.plot(df.index, df[f'{ticker} EMA_20'], label='EMA 20', color='red', linestyle='--')
        plt.title(f'{ticker} Price, SMA, EMA')
        plt.legend()
        plt.grid(True)

        # Subplot 2: MACD
        plt.subplot(5, 1, 2)
        plt.plot(df.index, df[f'{ticker} MACD'], label='MACD', color='purple')
        plt.plot(df.index, df[f'{ticker} Signal'], label='Signal Line', color='orange')
        plt.title(f'{ticker} MACD & Signal Line')
        plt.legend()
        plt.grid(True)

        # Subplot 3: RSI
        plt.subplot(5, 1, 3)
        plt.plot(df.index, df[f'{ticker} RSI'], label='RSI', color='green')
        plt.axhline(70, color='red', linestyle='--')
        plt.axhline(30, color='blue', linestyle='--')
        plt.title(f'{ticker} Relative Strength Index')
        plt.legend()
        plt.grid(True)

        # Subplot 4: Bollinger Bands
        plt.subplot(5, 1, 4)
        plt.plot(df.index, df[f'{ticker} Close'], label='Close Price', color='black')
        plt.plot(df.index, df[f'{ticker} BB_Upper'], label='Upper Band', linestyle='--', color='gray')
        plt.plot(df.index, df[f'{ticker} BB_Lower'], label='Lower Band', linestyle='--', color='gray')
        plt.fill_between(df.index, df[f'{ticker} BB_Lower'], df[f'{ticker} BB_Upper'], color='gray', alpha=0.1)
        plt.title(f'{ticker} Bollinger Bands')
        plt.legend()
        plt.grid(True)

        # Subplot 5: Volume and Volume % Change
        plt.subplot(5, 1, 5)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.bar(df.index, df[f'{ticker} Volume'], label='Volume', alpha=0.5, color='steelblue')
        ax2.plot(df.index, df[f'{ticker} Vol_Change'], label='Volume Change (%)', color='darkorange')
        ax1.set_ylabel('Volume')
        ax2.set_ylabel('Vol % Change')
        plt.title(f'{ticker} Volume and Volume % Change')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax1.grid(True)

        plt.tight_layout()
        plt.show()

def plot_decision_summary(decision_scores):

    print("- SMA (Simple Moving Average):")
    print("  • Smooths price data over time (e.g., 20 days).") 
    print("  • Buy: When price crosses above the SMA (uptrend start).")
    print("  • Sell: When price falls below the SMA (downtrend start).\n")
    
    print("- EMA (Exponential Moving Average):")
    print("  • Similar to SMA but more responsive to recent prices.")
    print("  • Buy: Price rises above EMA.")
    print("  • Sell: Price drops below EMA.\n")
    
    print("- Momentum:")
    print("  • Measures the rate of price change.")
    print("  • Buy: Positive momentum (price increasing).")
    print("  • Sell: Negative momentum (price falling).\n")
    
    print("- Stochastic Oscillator (%K, %D):")
    print("  • Compares closing price to price range over time.")
    print("  • Buy: Both %K and %D below 20 (oversold).")
    print("  • Sell: Both %K and %D above 80 (overbought).\n")
    
    print("- OBV (On Balance Volume):")
    print("  • Uses volume flow to predict price movement.")
    print("  • Buy: OBV rising with price (strong uptrend).")
    print("  • Sell: OBV falling with price (strong downtrend).\n")
    
    print("- ADX (Average Directional Index):")
    print("  • Measures trend strength (not direction).")
    print("  • Buy: ADX > 25 and uptrend confirmed by other indicators.")
    print("  • Sell: Weak trend or ADX < 20.\n")
    
    print("- RSI (Relative Strength Index):")
    print("  • Measures speed/size of price changes (0–100).")
    print("  • Buy: RSI < 30 (oversold).")
    print("  • Sell: RSI > 70 (overbought).\n")
    
    print("- MACD (Moving Average Convergence Divergence):")
    print("  • Trend-following momentum indicator.")
    print("  • Buy: MACD crosses above signal line.")
    print("  • Sell: MACD crosses below signal line.\n")
    
    print("- Bollinger Bands:")
    print("  • Bands widen/shrink based on volatility.")
    print("  • Buy: Price touches or breaks below lower band.")
    print("  • Sell: Price touches or breaks above upper band.\n")
    
    print("- Volume Change:")
    print("  • Measures how trading volume changes over time.")
    print("  • Buy: Increasing volume with price gains.")
    print("  • Sell: High volume with price drops.\n")

    df_scores = pd.DataFrame(decision_scores).T  # Transpose: stocks as rows
    df_scores = df_scores[['Buy', 'Hold', 'Sell']]  # Ensure consistent order

    color_map = ['green', 'gold', 'red']  # Buy, Hold, Sell
    df_scores.plot(kind='bar', stacked=True, figsize=(12, 6), color=color_map)

    plt.title("Buy / Hold / Sell Signal Distribution per Stock", fontsize=14)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.xlabel("Stock", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', title='Decision')
    plt.tight_layout()
    plt.show()

# def main():
#    ticker = ['PLTR','AMD','NVDA','META','GOOGL','ASML','MSFT','AVGO','IBM','TSLA','TSM']
#    start_date = "2010-10-01"
#    end_date = "2025-05-16" #y/m/d
#    api = 'SLNEQXVO3S7L9JTH' 
#    plot_on = True
#    tick = ['AMD']
  
#    data = donwload_data(ticker,start_date,end_date,api)

#    data  =  combine_data(ticker,start_date,end_date)

#    data, decisions = analyze_stock(ticker, data)

#    if plot_on:
#        visualize_stock(tick, data)

#    plot_decision_summary(decisions)


# if __name__ == '__main__':
#  main()
