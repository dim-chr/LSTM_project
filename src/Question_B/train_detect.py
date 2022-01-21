import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
import numpy as np
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.preprocessing import StandardScaler

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

num_time_series = 100

csv_path = os.path.join(os.path.abspath(__file__), "../../../dir/nasdaq2007_17.csv")

model_path = os.path.join(os.path.abspath(__file__), "../../../models/detect_model.h5")

df = pd.read_csv(csv_path, header=None, delimiter='\t')
file_ids = df.iloc[:, [0]].values
df = df.drop(df.columns[0], axis=1)
df = df.transpose()
print("Number of rows and columns:", df.shape)

TIME_STEPS = 30

# Training data: 95%, Test data: 5%
train_size = int(len(df) * 0.95)
test_size = len(df) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)

scaler = StandardScaler()

# Creating a data structure with 30 time-steps and 1 output
X_train = []
y_train = []
for step in range (0, num_time_series):
    
    training_set = df.iloc[:train_size, [step]].values
    training_set_scaled = scaler.fit_transform(training_set)

    for i in range(TIME_STEPS, train_size):
        X_train.append(training_set_scaled[i-TIME_STEPS:i, 0])
        y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape, y_train.shape)

model = Sequential()

model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(rate=0.2))

model.add(RepeatVector(n=X_train.shape[1]))

model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(rate=0.2))

model.add(TimeDistributed(Dense(units=X_train.shape[2])))

model.compile(loss='mae', optimizer='adam')

history = model.fit(X_train, y_train, epochs=10, batch_size=1024, validation_split=0.1, shuffle=False)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()
plt.clf()

# Save model
if os.path.isfile(model_path) is False:
    model.save(model_path)