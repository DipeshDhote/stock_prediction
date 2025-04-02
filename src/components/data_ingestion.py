import numpy as np
import pandas as pd
import yfinance as yf



# Step 1: Load Data
def load_data(stock_name, start_date='2015-01-01', end_date='2025-03-11'):
    df = yf.download(stock_name + ".NS", start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    return df