import os
import matplotlib.pyplot as plt
from numpy.core.multiarray import empty
import pandas as pd
import numpy as np
import sys
from keras.models import load_model
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler

if len(sys.argv) < 5:
    print("Too few arguments")
    quit()
elif len(sys.argv) > 5:
    print("Too many arguments")
    quit()

# Read the arguments
for i in range (1, len(sys.argv)):
    if(sys.argv[i] == "-d"):
        dataset_name = sys.argv[i+1]
    elif(sys.argv[i] == "-n"):
        num_time_series = int(sys.argv[i+1])

num = 100
csv_path = os.path.join(os.path.abspath(__file__), "../../../dir/")
csv_path = os.path.join(csv_path, dataset_name)

model_path = os.path.join(os.path.abspath(__file__), "../../../models/Forecast_all_model.h5")

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
X_train = []
y_train = []
for step in range (0, num):
    
    # Scale and concatenate all time series
    training_set = df.iloc[:train_size, [step]].values
    training_set_scaled = sc.fit_transform(training_set)

    for i in range(60, train_size):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Load model
model = load_model(model_path)

# For each time series get the prediction and create a plot
for series in range(0, num_time_series):
    
    dataset_train = df.iloc[:train_size, [series]]
    dataset_test = df.iloc[train_size:, [series]]
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)

    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, test_size+60):
        X_test.append(inputs[i-60:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    # Visualising the results
    plt.plot(dataset_test.values, color = 'red', label = 'Real values')
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted values')
    plt.title('Time Series Prediction')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(os.path.abspath(__file__), "../../../dir/exports/q1/var2/graph"+ str(series) +".png"))
    plt.clf()