import os
import matplotlib.pyplot as plt
from numpy.core.multiarray import empty
import pandas as pd
import numpy as np
import sys
import string
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler

if len(sys.argv) < 5:
    print("Too few arguments")
    quit()
elif len(sys.argv) > 5:
    print("Too many arguments")
    quit()

for i in range (1, len(sys.argv)):
    if(sys.argv[i] == "-d"):
        csv_path = sys.argv[i+1]
    elif(sys.argv[i] == "-n"):
        num = int(sys.argv[i+1])


curr_dir = os.path.abspath(__file__)
model_path = "../../Forecast_model.h5"
testpath = "/home/theo/Desktop/LSTM_project/dir/nasd_input.csv"

#1. append all -> train -> save image

df = pd.read_csv(testpath, header=None, delimiter='\t')
file_ids = df.iloc[:, [0]].values
df = df.drop(df.columns[0], axis=1)
df = df.transpose()
print("Number of rows and columns:", df.shape)

sc = MinMaxScaler(feature_range = (0, 1))    
# Creating a data structure with 60 time-steps and 1 output
X_train = []
y_train = []
for step in range (0 , num):
    
    training_set = df.iloc[:600, [step]].values  # Training set: first 800 rows
    training_set_scaled = sc.fit_transform(training_set)

    for i in range(60, 600):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Shape of X_train: (740, 60, 1)

model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Getting the predicted stock price of 2017
dataset_train = df.iloc[:600, 1:2]
dataset_test = df.iloc[600:, 1:2]
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 130):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test.shape)
# (459, 60, 1)

predicted_stock_price = model.predict(X_test)
print(len(predicted_stock_price))
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

print(len(dataset_test.values))
print(len(predicted_stock_price))

# Visualising the results
plt.plot(dataset_test.values, color = 'red', label = 'Real TESLA Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted TESLA Stock Price')
# plt.xticks(np.arange(0,459,50))
plt.title('TESLA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TESLA Stock Price')
plt.legend()
plt.show()
plt.savefig('q1a1.png')