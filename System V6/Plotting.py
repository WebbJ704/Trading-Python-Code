import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot(df, trades, **kwargs):

    MACD = kwargs.get('MACD',False)
    RSI = kwargs.get('RSI',False)
    RSI_low = kwargs.get('RSI_low',30)
    RSI_high = kwargs.get('RSI_high',70)
    RSI_window = kwargs.get('RSI_window',14)
    buy_sell = kwargs.get("buy_sell",False)
    buy_factor = kwargs.get("buy_factor",0.95)
    sell_factor = kwargs.get("sell_factor",1.1)
    BB = kwargs.get("BB",False)


    if MACD:
        trade_signals = df[df['Signal'] == 1]
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        # Price + EMAs + Signals (Top)
        ax1.plot(df['Close'], label='Close Price', color='black')
        ax1.plot(df['EMA_short'], label='EMA short', color='orange')
        ax1.plot(df['EMA_long'], label='EMA long', color='blue')
        ax1.scatter(trade_signals.index, trade_signals['Close'], color='green', label='Signal', zorder=5)
        ax1.set_title("Trade Signals")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.tick_params(axis='x', rotation=0)
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.tick_params(axis='x', rotation=45)
        # MACD (Bottom) 
        ax2.plot(df['MACD_Signal'], label='MACD signal', color='gold')
        ax2.plot(df['MACD'], label='MACD', color='green')
        ax2.set_title("MACD")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("MACD")
        ax2.legend()
        ax2.tick_params(axis='x', rotation=0)
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
    
    if RSI:
        trade_signals = df[df['Signal'] == 1]
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        # Price + EMAs + Signals (Top)
        ax1.plot(df['Close'], label='Adj Close Price', color='black')
        ax1.scatter(trade_signals.index, trade_signals['Close'], color='green', label='Signal', zorder=5)
        ax1.set_title("Trade Signals")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.tick_params(axis='x', rotation=0)
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.tick_params(axis='x', rotation=45)
        # RSI (Bottom)
        ax2.plot(df.index, df['RSI'], label=f'RSI({RSI_window})', color='purple')
        ax2.axhline(RSI_high, linestyle='--', color='red', alpha=0.7)
        ax2.axhline(RSI_low, linestyle='--', color='green', alpha=0.7)
        ax2.set_title(f"Relative Strength Index (RSI({RSI_window}))")
        ax2.set_xlabel("Date")
        ax2.set_ylabel(f"RSI({RSI_window})")
        ax2.legend()
        ax2.tick_params(axis='x', rotation=0)
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 100)
        plt.tight_layout()
        plt.show()
    
    if buy_sell:
        trade_signals = df[df['Signal'] == 1]
        fig, ax = plt.subplots(figsize=(14, 8))
        # Price + EMAs + Signals (Top)
        ax.plot(df['Close'], label='Adj Close Price', color='black')
        ax.scatter(trade_signals.index, trade_signals['Close'], color='green', label='Signal', zorder=5)
        ax.set_title(f"Trade Signals for buy factor {buy_factor} and sell factor {sell_factor}")
        ax.set_ylabel("Price")
        ax.legend()
        ax.tick_params(axis='x', rotation=0)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=45)

    if BB:
        trade_signals = df[df['Signal'] == 1]
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Price + BB Bands + Signals (Top)
        ax1.plot(df.index, df['Close'], label='Close Price', color='black')
        ax1.plot(df.index, df['BB_Middle'], label='BB middle', color='orange')   # fixed 'middel' to 'middle'
        ax1.plot(df.index, df['BB_Upper'], linestyle='--', label='BB upper', color='blue')
        ax1.plot(df.index, df['BB_Lower'], linestyle='--', label='BB lower', color='cyan')  # different color
        
        ax1.scatter(trade_signals.index, trade_signals['Close'], color='green', label='Signal', zorder=5)
        
        ax1.set_title("Trade Signals")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Cumulative returns (Bottom)
        ax2.plot(trades.index, trades['StrategyEquity'], label='Strategy')
        ax2.plot(trades.index, trades['BuyAndHold'], label='Buy & Hold', linestyle='--')
        
        ax2.set_title("Cumulative Returns Comparison")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Return")  # fixed typo
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.tight_layout()
        plt.show()

    # Ceeate figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot the strategy and buy & hold
    ax.plot(trades.index, trades['StrategyEquity'], label='Strategy')
    ax.plot(trades.index, trades['BuyAndHold'], label='Buy & Hold', linestyle='--')
    # Format the x-axis as datetime
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.tick_params(axis='x', rotation=45)
    # Add titles and labels
    ax.set_title('Cumulative Returns Comparison')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()