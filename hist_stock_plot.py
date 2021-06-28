import datetime
import pandas as pd
import matplotlib.pyplot as plt

today = datetime.datetime.today()
end_date = str(int(today.timestamp()))
time_delta = datetime.timedelta(days = 60)
start_date = str(int((today - time_delta).timestamp()))


stock_code = input("Enter Stock Code: ").upper()
stock_interval = '1d'
stock_events = 'history'

csv_url = "https://query1.finance.yahoo.com/v7/finance/download/" + stock_code + ".NS?period1=" + start_date + "&period2=" + end_date + "&interval=" + stock_interval + "&events=" + stock_events + "&includeAdjustedClose=true"

stock_data = pd.read_csv(csv_url)
# Remove rows with Nan values
stock_data = stock_data.dropna()
print(stock_data)

ax = plt.gca()
stock_data.plot(kind = 'line', x = 'Date', y = 'Close', ax = ax)
stock_data.plot(kind = 'line', x = 'Date', y = 'High', ax = ax, color = 'green')
stock_data.plot(kind = 'line', x = 'Date', y = 'Low', ax = ax, color = 'red')
plt.show()