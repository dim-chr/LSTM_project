from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model
from keras.models import model_from_json
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# %pylab inline

window_length = 10
encoding_dim = 3
epochs = 100

csv_path = os.path.join(os.path.abspath(__file__), "../../../dir/nasdaq2007_17.csv")
num_time_series = 100

encoder_path = os.path.join(os.path.abspath(__file__), "../../../models/encoder.h5")
autoencoder_path = os.path.join(os.path.abspath(__file__), "../../../models/autoencoder.h5")

df = pd.read_csv(csv_path, header=None, delimiter='\t')
file_ids = df.iloc[:, [0]].values
df = df.drop(df.columns[0], axis=1)
df = df.transpose()

# Training data: 90%, Test data: 10%
train_size = int(len(df) * 0.90)
test_size = len(df) - train_size

scaler = MinMaxScaler()

# Creating a data structure with window_length 10
x_train = []
for step in range (0, num_time_series):
    
    training_set = df.iloc[:train_size, [step]].values
    training_set_scaled = scaler.fit_transform(training_set)

    for i in range(window_length, train_size):
        x_train.append(training_set_scaled[i-window_length:i, 0])

x_train = np.array(x_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

dataset_train = df.iloc[:train_size, [0]]
dataset_test = df.iloc[train_size:, [0]]
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - window_length:].values
inputs = inputs.reshape(-1,1)
print("Input shape: ", inputs.shape)
inputs = scaler.transform(inputs)
x_test = []
for i in range(window_length, test_size+window_length):
    x_test.append(inputs[i-window_length:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print(x_test.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

input_window = Input(shape=(window_length,1))
x = Conv1D(16, 3, activation="relu", padding="same")(input_window) # 10 dims
#x = BatchNormalization()(x)
x = MaxPooling1D(2, padding="same")(x) # 5 dims
x = Conv1D(1, 3, activation="relu", padding="same")(x) # 5 dims
#x = BatchNormalization()(x)
encoded = MaxPooling1D(2, padding="same")(x) # 3 dims

encoder = Model(input_window, encoded)

# 3 dimensions in the encoded layer

x = Conv1D(1, 3, activation="relu", padding="same")(encoded) # 3 dims
x = BatchNormalization()(x)
x = UpSampling1D(2)(x) # 6 dims
x = Conv1D(16, 2, activation='relu')(x) # 5 dims
#x = BatchNormalization()(x)
x = UpSampling1D(2)(x) # 10 dims
decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x) # 10 dims
autoencoder = Model(input_window, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history = autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=1024,
                shuffle=True,
                validation_data=(x_test, x_test))

encoder.compile(optimizer='adam', loss='binary_crossentropy')

# Save models
if os.path.isfile(autoencoder_path) is False:
    autoencoder.save(autoencoder_path)
    encoder.save(encoder_path)