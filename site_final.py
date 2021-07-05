import math
import datetime
import streamlit as st
import numpy as np
import pandas as pd
from plotly import graph_objs as go
import matplotlib.pyplot as plt
from alpha_vantage.techindicators import TechIndicators
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

st.set_page_config(page_title = 'Stock Prediction App')
st.markdown(
f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 1000px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }}
</style>
""", unsafe_allow_html=True,
)
st.title('Standard Wings Technology - Stock Prediction App')
st.write('An app created with Python to analyze and predict the stock data.')
st.markdown('- - - -')

@st.cache
def load_data(stock_code):
	today = datetime.datetime.today()
	end_date = str(int(today.timestamp()))
	time_delta = datetime.timedelta(days = 365)
	start_date = str(int((today - time_delta).timestamp()))
	stock_interval = '1d'
	stock_events = 'history'
	csv_url = 'https://query1.finance.yahoo.com/v7/finance/download/' + stock_code + '?period1=' + start_date + '&period2=' + end_date + '&interval=' + stock_interval + '&events=' + stock_events + '&includeAdjustedClose=true'
	stock_data = pd.read_csv(csv_url)
	# Remove rows with Nan values
	stock_data = stock_data.dropna()

	# For Loading Moving Avgs Data
	ema_dataframe = stock_data.filter(['Date', 'Close'])
	ema_dataframe = ema_dataframe.set_index(pd.DatetimeIndex(ema_dataframe['Date'].values))
	# Short exponential moving average
	short_EMA = ema_dataframe.Close.ewm(span = 5, adjust = False).mean()
	# Medium exponential moving average
	medium_EMA = ema_dataframe.Close.ewm(span = 20, adjust = False).mean()
	# Long exponential moving average
	long_EMA = ema_dataframe.Close.ewm(span = 60, adjust = False).mean()
	ema_dataframe['Short EMA'] = short_EMA
	ema_dataframe['Medium EMA'] = medium_EMA
	ema_dataframe['Long EMA'] = long_EMA
	ema_dataframe = ema_dataframe.reset_index(drop = True)

	return stock_data, ema_dataframe

selected_stock = st.text_input('Enter Stock Code:').upper()

if st.button('Submit'):
	data_load_state = st.text('Loading Data...')
	raw_dataframe, ema_dataframe = load_data(selected_stock)
	data_load_state.text('')

	st.subheader('Raw Data:')
	st.dataframe(raw_dataframe, 900, 400)

	def plot_raw_data():
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=raw_dataframe['Date'], y=raw_dataframe['Open'], name='Open'))
		fig.add_trace(go.Scatter(x=raw_dataframe['Date'], y=raw_dataframe['High'], name='High'))
		fig.add_trace(go.Scatter(x=raw_dataframe['Date'], y=raw_dataframe['Low'], name='Low'))
		fig.add_trace(go.Scatter(x=raw_dataframe['Date'], y=raw_dataframe['Close'], name='Close'))
		
		fig.update_layout(
			autosize = False,
			width = 900,
			height = 500,
			margin = dict(l = 0, r = 0, b = 0, t = 0, pad = 0),
			yaxis = dict(title_text = 'Price', titlefont = dict(size = 18)),
			xaxis = dict(title_text = 'Date', titlefont = dict(size = 18)),
		)
		st.subheader('Time Series Data with Rangeslider:')
		fig.layout.update(xaxis_rangeslider_visible=True)
		st.plotly_chart(fig)
	plot_raw_data()

	st.subheader('Exponential Moving Averages:')
	st.subheader('Data for EMA:')
	st.dataframe(ema_dataframe, 900, 400)

	def plot_ema_data():
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=ema_dataframe['Date'], y=ema_dataframe['Close'], name='Close Price'))
		fig.add_trace(go.Scatter(x=ema_dataframe['Date'], y=ema_dataframe['Short EMA'], name='Short EMA'))
		fig.add_trace(go.Scatter(x=ema_dataframe['Date'], y=ema_dataframe['Medium EMA'], name='Medium EMA'))
		fig.add_trace(go.Scatter(x=ema_dataframe['Date'], y=ema_dataframe['Long EMA'], name='Long EMA'))
		fig.update_layout(
			autosize = False,
			width = 900,
			height = 500,
			margin = dict(l = 0, r = 0, b = 0, t = 0, pad = 0),
			yaxis = dict(title_text = 'Price', titlefont = dict(size = 18)),
			xaxis = dict(title_text = 'Date', titlefont = dict(size = 18)),
		)
		st.subheader('Plot for EMA:')
		fig.layout.update(xaxis_rangeslider_visible=True)
		st.plotly_chart(fig)
	plot_ema_data()

	# SMA and RSI
	st.subheader('Simple Moving Average and RSI:')
	api_key = 'RNZPXZ6Q9FEFMEHM'
	period = 60

	ti = TechIndicators(key=api_key, output_format='pandas')
	data_ti, meta_data_ti = ti.get_rsi(symbol=selected_stock, interval='1min', time_period=period, series_type='close')
	data_sma, meta_data_sma = ti.get_sma(symbol=selected_stock, interval='1min', time_period=period, series_type='close')

	df1 = data_sma.iloc[1::]
	df2 = data_ti
	df1.index = df2.index

	fig, ax1 = plt.subplots()
	ax1.plot(df1, 'b-')
	ax2 = ax1.twinx()
	ax2.plot(df2, 'r.')
	plt.title("SMA & RSI graph")
	plt.figure(figsize=(4, 3))
	plt.setp(ax1.get_xticklabels(), fontsize = 8, rotation = 40, horizontalalignment = 'right')
	st.pyplot(fig)

	# Sentiment Analysis :
	st.subheader('Sentiment Analysis :')

	finviz_url = 'https://finviz.com/quote.ashx?t='
	ticker = selected_stock

	news_tables = {}

	url = finviz_url + ticker
	req = Request(url=url, headers={'user-agent': 'my-app'})
	response = urlopen(req)

	html = BeautifulSoup(response, 'lxml')
	news_table = html.find(id='news-table')
	news_tables[ticker] = news_table

	ticker_data = news_tables[ticker]
	ticker_rows = ticker_data.findAll('tr')

	for index, row in enumerate(ticker_rows):
		title = row.a.text
		timestamp = row.td.text

	parsed_data = []
	
	for row in news_table.findAll('tr'):
		title = row.a.text
		date_data = row.td.text.split(' ')

		if len(date_data) == 1:
			time = date_data[0]
		else:
			date = date_data[0]
			time = date_data[1]
			
		parsed_data.append([ticker, date, time, title])

	df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

	vader = SentimentIntensityAnalyzer()

	f = lambda title: vader.polarity_scores(title)['compound']

	# adding 'compound' column
	df['compound'] = df['title'].apply(f)
	df['date'] = pd.to_datetime(df.date).dt.date
	
	plt.rcParams['figure.figsize'] = [4, 3]
	mean_df = df.groupby(['ticker', 'date']).mean()
	mean_df = mean_df.unstack()
	mean_df = mean_df.xs('compound', axis="columns").transpose()
	mean_df.plot(kind='bar')
	plt.title('Sentiment Graph')
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot()

	# Prediction using LSTM
	st.subheader('Using LSTM Model to Predict Close Prices:')
	today = datetime.datetime.today()
	lstm_train_end_date = str(int(today.timestamp()))
	time_delta = datetime.timedelta(days = 900)
	lstm_train_start_date = str(int((today - time_delta).timestamp()))

	lstm_stock_code = selected_stock
	lstm_stock_interval = '1d'
	lstm_stock_events = 'history'

	lstm_csv_url = "https://query1.finance.yahoo.com/v7/finance/download/" + lstm_stock_code + "?period1=" + lstm_train_start_date + "&period2=" + lstm_train_end_date + "&interval=" + lstm_stock_interval + "&events=" + lstm_stock_events + "&includeAdjustedClose=true"
	lstm_train_stock_data = pd.read_csv(lstm_csv_url)
	# Remove null values
	lstm_train_stock_data = lstm_train_stock_data.dropna()
	# Only Close price column
	lstm_close_data = lstm_train_stock_data.filter(['Close'])
	# Convert to numpy array
	train_dataset = lstm_close_data.values
	# Get number of rows to train the model on
	training_data_len = math.ceil(len(train_dataset) * .8) # 251rowsx7cols = 201 for SBIN
	# Scale the data
	scaler = MinMaxScaler(feature_range=(0,1))
	scaled_data = scaler.fit_transform(train_dataset)

	# Create the scaled training data set
	train_data = scaled_data[0:training_data_len , :]

	# Split data into x_train and y_train data sets
	x_train = []
	y_train = []

	for i in range(60, len(train_data)):
		x_train.append(train_data[i-60:i, 0])
		y_train.append(train_data[i, 0])

	# Convert x_train and y_train into numpy arrays
	x_train, y_train = np.array(x_train), np.array(y_train)

	# Reshape the data
	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

	# Build the LSTM Model
	model = Sequential()
	model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
	model.add(LSTM(50, return_sequences = False))
	model.add(Dense(25))
	model.add(Dense(1))

	# Compile the Model
	model.compile(optimizer = 'adam', loss = 'mean_squared_error')

	# train the model
	model.fit(x_train, y_train, batch_size = 1, epochs = 1)

	# Creating testing data set
	# Create a new array containing scaled values
	test_data = scaled_data[training_data_len - 60:, :]

	# Creating the data sets x_test and y_test
	x_test = []
	y_test = train_dataset[training_data_len:, :]
	for i in range(60, len(test_data)):
		x_test.append(test_data[i-60:i, 0])

	# Convert data into numpy array
	x_test = np.array(x_test)

	# Reshape data
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

	# Get the models predicted price values
	predictions = model.predict(x_test)
	# Unscaling the values
	predictions = scaler.inverse_transform(predictions)

	# Get the root mean squared error (RMSE) - measure of how accurate it is, lower means better 
	rmse=np.sqrt(np.mean(((predictions- y_test)**2)))

	# Plot the data
	train = lstm_close_data[:training_data_len]
	valid = lstm_close_data[training_data_len:]
	valid['Predictions'] = predictions

	# Visualize
	st.subheader('Predicted Data:')
	st.dataframe(valid, 900, 400)

	st.subheader('Plot for Predictions:')
	st.write('Close Price Trained on')
	lstm_fig1 = go.Figure()
	lstm_fig1.add_trace(go.Scatter(y=train['Close'], name='Close Price Trained on'))
	lstm_fig1.update_layout(
		autosize = False,
		width = 900,
		height = 500,
		margin = dict(l = 0, r = 0, b = 0, t = 0, pad = 0),
		yaxis = dict(title_text = 'Price', titlefont = dict(size = 18)),
		xaxis = dict(title_text = 'No. of Days', titlefont = dict(size = 18)),
	)
	st.plotly_chart(lstm_fig1)
	st.write('Actual Price versus Predictions:')
	lstm_fig2 = go.Figure()
	lstm_fig2.add_trace(go.Scatter(y=valid['Close'], name='Actual Price'))
	lstm_fig2.add_trace(go.Scatter(y=valid['Predictions'], name='Predictions'))
	lstm_fig2.update_layout(
		autosize = False,
		width = 900,
		height = 500,
		margin = dict(l = 0, r = 0, b = 0, t = 0, pad = 0),
		yaxis = dict(title_text = 'Price', titlefont = dict(size = 18)),
		xaxis = dict(title_text = 'No. of Days', titlefont = dict(size = 18)),
	)
	st.plotly_chart(lstm_fig2)
	st.write("RMSE Value: ", rmse)