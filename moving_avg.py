import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

today = datetime.datetime.today()
end_date = str(int(today.timestamp()))
time_delta = datetime.timedelta(days = 365)
start_date = str(int((today - time_delta).timestamp()))

stock_code = input("Enter Stock Code: ").upper()
stock_interval = '1d'
stock_events = 'history'
csv_url = "https://query1.finance.yahoo.com/v7/finance/download/" + stock_code + "?period1=" + start_date + "&period2=" + end_date + "&interval=" + stock_interval + "&events=" + stock_events + "&includeAdjustedClose=true"
stock_data = pd.read_csv(csv_url)
# Remove rows with Nan values
stock_data = stock_data.dropna()
print(stock_data)

ma_dataframe = stock_data.filter(['Date', 'Close'])
ma_dataframe = ma_dataframe.set_index(pd.DatetimeIndex(ma_dataframe['Date'].values))
# Short exponential moving average
short_EMA = ma_dataframe.Close.ewm(span = 5, adjust = False).mean()
# Medium exponential moving average
medium_EMA = ma_dataframe.Close.ewm(span = 20, adjust = False).mean()
# Long exponential moving average
long_EMA = ma_dataframe.Close.ewm(span = 60, adjust = False).mean()

ma_dataframe['Short EMA'] = short_EMA
ma_dataframe['Medium EMA'] = medium_EMA
ma_dataframe['Long EMA'] = long_EMA

print(ma_dataframe)

plt.figure(figsize=(16,9))
plt.title('Close Price with Exponential Moving Average')
plt.xlabel('Date')
plt.ylabel('Close Price (INR)')
plt.plot(ma_dataframe['Close'], label = 'Close Price', color = 'deepskyblue')
plt.plot(short_EMA, label = 'Short EMA', color = 'red')
plt.plot(medium_EMA, label = 'Medium EMA', color = 'orange')
plt.plot(long_EMA, label = 'Long EMA', color = 'lightgreen')
plt.legend(['Close Price', 'Short EMA', 'Medium EMA', 'Long EMA'], loc = 'upper left')
plt.show()