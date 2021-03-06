import datetime
import pandas as pd

start_date = str(int(datetime.datetime(2020, 11, 12).timestamp()))
end_date = str(int(datetime.datetime(2020, 11, 22).timestamp()))
stock_code = 'SBIN'
stock_interval = '1d'
stock_events = 'history'

csv_url = "https://query1.finance.yahoo.com/v7/finance/download/" + stock_code + ".NS?period1=" + start_date + "&period2=" + end_date + "&interval=" + stock_interval + "&events=" + stock_events + "&includeAdjustedClose=true"

stock_data = pd.read_csv(csv_url)
stock_data = stock_data.dropna()
print(stock_data)