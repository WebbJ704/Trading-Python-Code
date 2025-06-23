import data_download as dataDownlaod
import Rules as rules
# ----------------- CONFIG -----------------
SYMBOL = 'AAPL'
# ------------------------------------------
if __name__ == "__main__" :
    df = dataDownlaod.fetch_yf_data(SYMBOL)
    df = rules.Rules(df)
    print(df.head())