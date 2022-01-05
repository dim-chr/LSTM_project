import os
import math
import matplotlib.pyplot as plt
import keras
from numpy.lib.npyio import load
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler

curr_dir = os.path.abspath(__file__)
csv_path = "..\..\dir\TSLA.csv"
model_path = "..\..\Forecast_model.h5"

df = pd.read_csv(os.path.join(curr_dir, csv_path))
print("Number of rows and columns:", df.shape)

model = load_model(os.path.join(curr_dir, model_path))
training_set = df.iloc[:800, [1]].values  # Training set: first 800 rows

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Getting the predicted stock price of 2017
dataset_train = df.iloc[:800, 1:2]
dataset_test = df.iloc[800:, 1:2]
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 519):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test.shape)
# (459, 60, 1)

predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(df.loc[800:, 'Date'],dataset_test.values, color = 'red', label = 'Real TESLA Stock Price')
plt.plot(df.loc[800:, 'Date'],predicted_stock_price, color = 'blue', label = 'Predicted TESLA Stock Price')
plt.xticks(np.arange(0,459,50))
plt.title('TESLA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TESLA Stock Price')
plt.legend()
plt.show()