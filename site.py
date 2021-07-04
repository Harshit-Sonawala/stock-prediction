import datetime
from pathlib import WindowsPath
import streamlit as st
import numpy as np
import pandas as pd
from plotly import graph_objs as go
import matplotlib.pyplot as plt
from alpha_vantage.techindicators import TechIndicators

st.set_page_config(page_title = 'Stock Prediction App', layout = 'wide')
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

	col1, col2 = st.beta_columns((1,2))
	col1.subheader('Raw Data:')
	col1.dataframe(raw_dataframe, 800, 500)

	def plot_raw_data():
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=raw_dataframe['Date'], y=raw_dataframe['Open'], name='Open'))
		fig.add_trace(go.Scatter(x=raw_dataframe['Date'], y=raw_dataframe['High'], name='High'))
		fig.add_trace(go.Scatter(x=raw_dataframe['Date'], y=raw_dataframe['Low'], name='Low'))
		fig.add_trace(go.Scatter(x=raw_dataframe['Date'], y=raw_dataframe['Close'], name='Close'))
		
		fig.update_layout(
			autosize = False,
			width = 1200,
			height = 500,
			margin = dict(l = 0, r = 0, b = 0, t = 0, pad = 0),
			yaxis = dict(title_text = 'Price', titlefont = dict(size = 18)),
			xaxis = dict(title_text = 'Date', titlefont = dict(size = 18)),
		)
		col2.subheader('Time Series Data with Rangeslider:')
		fig.layout.update(xaxis_rangeslider_visible=True)
		col2.plotly_chart(fig)
	plot_raw_data()

	col1.subheader('Exponential Moving Averages:')
	col1.subheader('Data for EMA:')
	col1.dataframe(ema_dataframe, 800, 500)

	def plot_ema_data():
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=ema_dataframe['Date'], y=ema_dataframe['Close'], name='Close Price'))
		fig.add_trace(go.Scatter(x=ema_dataframe['Date'], y=ema_dataframe['Short EMA'], name='Short EMA'))
		fig.add_trace(go.Scatter(x=ema_dataframe['Date'], y=ema_dataframe['Medium EMA'], name='Medium EMA'))
		fig.add_trace(go.Scatter(x=ema_dataframe['Date'], y=ema_dataframe['Long EMA'], name='Long EMA'))
		fig.update_layout(
			autosize = False,
			width = 1200,
			height = 500,
			margin = dict(l = 0, r = 0, b = 0, t = 0, pad = 0),
			yaxis = dict(title_text = 'Price', titlefont = dict(size = 18)),
			xaxis = dict(title_text = 'Date', titlefont = dict(size = 18)),
		)
		col2.subheader('Plot for EMA:')
		fig.layout.update(xaxis_rangeslider_visible=True)
		col2.plotly_chart(fig)
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
	plt.figure(figsize=(2, 3))
	st.pyplot(fig)

