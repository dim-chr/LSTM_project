import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
import numpy as np
import seaborn as sns
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

if len(sys.argv) < 7:
    print("Too few arguments")
    quit()
elif len(sys.argv) > 7:
    print("Too many arguments")
    quit()

# Read the arguments
for i in range (1, len(sys.argv)):
    if(sys.argv[i] == "-d"):
        dataset_name = sys.argv[i+1]
    elif(sys.argv[i] == "-n"):
        num_time_series = int(sys.argv[i+1])
    elif(sys.argv[i] == "-mae"):
        mae = float(sys.argv[i+1])

num = 100
csv_path = os.path.join(os.path.abspath(__file__), "../../../dir/")
csv_path = os.path.join(csv_path, dataset_name)

model_path = os.path.join(os.path.abspath(__file__), "../../../models/detect_model.h5")

df = pd.read_csv(csv_path, header=None, delimiter='\t')
file_ids = df.iloc[:, [0]].values
df = df.drop(df.columns[0], axis=1)
df = df.transpose()

TIME_STEPS = 30

# Training data: 95%, Test data: 5%
train_size = int(len(df) * 0.95)
test_size = len(df) - train_size

scaler = StandardScaler()

# Creating a data structure with 30 time-steps and 1 output
X_train = []
y_train = []
for step in range (0, num):
    
    training_set = df.iloc[:train_size, [step]].values
    training_set_scaled = scaler.fit_transform(training_set)

    for i in range(TIME_STEPS, train_size):
        X_train.append(training_set_scaled[i-TIME_STEPS:i, 0])
        y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

dataset_test = df.iloc[train_size:, [0]]

# Load model
model = load_model(model_path)

X_train_pred = model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

for series in range(0, num_time_series):
    dataset_train = df.iloc[:train_size, [series]]
    dataset_test = df.iloc[train_size:, [series]]
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)

    inputs = dataset_total[len(dataset_total) - len(dataset_test) - TIME_STEPS:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)
    X_test = []
    for i in range(TIME_STEPS, test_size):
        X_test.append(inputs[i-TIME_STEPS:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    X_test_pred = model.predict(X_test)
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

    THRESHOLD = mae

    # DataFrame containing the loss and the anomalies (values above the threshold)
    test_score_df = pd.DataFrame(index=dataset_test[TIME_STEPS:].index)
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    test_score_df['value'] = dataset_test[TIME_STEPS:]

    anomalies = test_score_df[test_score_df.anomaly == True]

    plt.plot( 
        dataset_test[TIME_STEPS:].index, 
        dataset_test[TIME_STEPS:],
        label='value')

    sns.scatterplot(
        anomalies.index, 
        anomalies.value, 
        color=sns.color_palette()[3], 
        s=52, 
        label='anomaly')

    plt.xticks(rotation=25)
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(os.path.abspath(__file__), "../../../dir/exports/q2/graph"+ str(series) +".png"))
    plt.clf()