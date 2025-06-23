import single_stock_analysis_code as sm
import pandas as pd
import tkinter as tk
from tkinter import filedialog
def main():

    # Open the file selection dialog
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select a file", filetypes=(("Text Files", "*.txt"), ("All Files", "*.*"))) 

    # reads File
    df = pd.read_csv(file_path,  parse_dates=['start_date', 'end_date'])
    ticker = str(df['ticker'].iloc[0])
    start_date =df['start_date'].dt.strftime('%Y-%m-%d').iloc[0]
    end_date = df['end_date'].dt.strftime('%Y-%m-%d').iloc[0]
    api = 'SLNEQXVO3S7L9JTH'
    return_window = df['retrun_window'].iloc[0]
    RSI_window = df['RSI_window'].iloc[0]
    short_window = df['short_window'].iloc[0]
    long_window = df['Long_window'].iloc[0]
    RSI_overbought = df['RSI_overbought'].iloc[0]
    RSI_underbought = df['RSI_underbought'].iloc[0]
    adx_window = df['ADX Window'].iloc[0]
    anual_window = df['Anual Window'].iloc[0]

    # Calualtions/plots from  Self made 
    data  =  sm.data_download_pl(ticker,api,start_date,end_date)

    data = sm.returns(data,return_window)

    std_dev , anual_volatitly , data, low_volatility ,high_volatility = sm.standard_deviation(data,return_window)

    mean_results = sm.mean(data)
    
    x,y = sm.prob_den_fs(mean_results,std_dev)

    data = sm.calculate_adx(data,adx_window)

    data = sm.RSI(data,RSI_window,RSI_overbought,RSI_underbought)

    data = sm.moving_averages(data,short_window,long_window)
    
    data = sm.calculate_cumulative_return(data)

    data = sm.Sharp_ratio(data,anual_window)

    sm.plotting(data,ticker,return_window,std_dev,x,y,mean_results,std_dev,anual_volatitly,RSI_window,short_window,long_window, low_volatility ,high_volatility)
    
    
if __name__ == '__main__':
 main()