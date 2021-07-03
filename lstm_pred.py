import math
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
plt.style.use('fivethirtyeight')

today = datetime.datetime.today()
end_date = str(int(today.timestamp()))
time_delta = datetime.timedelta(days = 365)
start_date = str(int((today - time_delta).timestamp()))


stock_code = input("Enter Stock Code: ").upper()
stock_interval = '1d'
stock_events = 'history'

csv_url = "https://query1.finance.yahoo.com/v7/finance/download/" + stock_code + ".NS?period1=" + start_date + "&period2=" + end_date + "&interval=" + stock_interval + "&events=" + stock_events + "&includeAdjustedClose=true"
print(csv_url)
stock_data = pd.read_csv(csv_url)
# Remove null values
stock_data = stock_data.dropna()
print(stock_data)
print("\n\n")

# Only Close price column
data = stock_data.filter(['Close'])
# Convert to numpy array
dataset = data.values
# Get number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8) # 251rowsx7cols = 201 for SBIN
# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

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
y_test = dataset[training_data_len:, :]
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
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Visualize
plt.figure(figsize=(14,6))
plt.title(f"Model for {stock_code}:")
plt.xlabel('Number of Days', fontsize = 18)
plt.ylabel('Close Price (INR)', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Actual Value', 'Predictions'], loc = 'lower right')
plt.show()

print(valid)
print("\n\nRMSE Value: ", rmse)