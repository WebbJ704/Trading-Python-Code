import analysis as A

def main():
   ticker = ['PLTR','AMD','NVDA','META','GOOGL','ASML','MSFT','AVGO','IBM','TSLA','TSM']
   start_date = "2010-10-01"
   end_date = "2025-05-21" #y/m/d
   api = '' 
   plot_on = True
   tick = ['AMD']
  
   data = A.donwload_data(ticker,start_date,end_date,api)

   data  =  A.combine_data(ticker,start_date,end_date)

   data, decisions = A.analyze_stock(ticker, data)

   if plot_on:
       A.visualize_stock(tick, data)

   A.plot_decision_summary(decisions)


if __name__ == '__main__':
 main()
