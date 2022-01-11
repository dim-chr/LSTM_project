import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.preprocessing import StandardScaler

TIME_STEPS = 30
THRESHOLD = 0.65

curr_dir = os.path.abspath(__file__)
csv_path = "../../../dir/spx.csv"
model_path = "../../Forecast_model.h5"
testpath = "/home/theo/Desktop/LSTM_project/dir/nasd_input.csv"

df = pd.read_csv(os.path.join(curr_dir, csv_path))
df = df.drop(df.columns[0], axis=1)

# Training data: 95%
train_size = int(len(df) * 0.95)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

scaler = StandardScaler()
scaler = scaler.fit(train[['close']])

train['close'] = scaler.transform(train[['close']])
test['close'] = scaler.transform(test[['close']])

# Creating a data structure with 30 time-steps and 1 output
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


# reshape to [samples, time_steps, n_features]
X_train, y_train = create_dataset(train[['close']], train.close, TIME_STEPS)

X_test, y_test = create_dataset(test[['close']], test.close, TIME_STEPS)

print(X_train.shape)

model = Sequential()

model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(rate=0.2))

model.add(RepeatVector(n=X_train.shape[1]))

model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(rate=0.2))

model.add(TimeDistributed(Dense(units=X_train.shape[2])))

model.compile(loss='mae', optimizer='adam')

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, shuffle=False)

X_train_pred = model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

X_test_pred = model.predict(X_test)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

# DataFrame containing the loss and the anomalies (values above the threshold)
test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['close'] = test[TIME_STEPS:].close

anomalies = test_score_df[test_score_df.anomaly == True]