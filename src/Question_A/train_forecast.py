import os
import pandas as pd
import numpy as np
from keras.backend import clear_session
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler

num_time_series = 2

csv_path = os.path.join(os.path.abspath(__file__), "../../../dir/nasdaq2007_17.csv")

# Read the input file
df = pd.read_csv(csv_path, header=None, delimiter='\t')
file_ids = df.iloc[:, [0]].values
df = df.drop(df.columns[0], axis=1)
df = df.transpose()

# Training data: 80%, Test data: 20%
train_size = int(len(df) * 0.80)
test_size = len(df) - train_size

sc = MinMaxScaler(feature_range = (0, 1))

# Creating a data structure with 60 time-steps and 1 output
for step in range (0, num_time_series):
    X_train = []
    y_train = []
    
    training_set = df.iloc[:train_size, [step]].values
    training_set_scaled = sc.fit_transform(training_set)

    for i in range(60, train_size):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Train a different model for each time series
    model = Sequential()
    
    #Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
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
    model.fit(X_train, y_train, epochs = 100, batch_size = 128, validation_split=0.1)

    # Save model
    if os.path.isfile(os.path.join(os.path.abspath(__file__), "../../../models/Forecast_model"+ str(step) +".h5")) is False:
        model.save(os.path.join(os.path.abspath(__file__), "../../../models/Forecast_model"+ str(step) +".h5"))
    
    del model
    clear_session()