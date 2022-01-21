from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model
from keras.models import model_from_json
from keras.models import load_model
import pandas as pd
import os
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# %pylab inline

if len(sys.argv) < 9:
    print("Too few arguments")
    quit()
elif len(sys.argv) > 9:
    print("Too many arguments")
    quit()

for i in range (1, len(sys.argv)):
    if(sys.argv[i] == "-d"):
        input_file = sys.argv[i+1]
    elif(sys.argv[i] == "-q"):
        query_file = sys.argv[i+1]
    elif(sys.argv[i] == "-od"):
        out_input_file = sys.argv[i+1]
    elif(sys.argv[i] == "-oq"):
        out_query_file = sys.argv[i+1]

window_length = 10
encoding_dim = 3
epochs = 100
test_samples = 365

def plot_examples(stock_input, stock_decoded):
    n = 10  
    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(list(np.arange(0, test_samples, 50))):
        # display original
        ax = plt.subplot(2, n, i + 1)
        if i == 0:
            ax.set_ylabel("Input", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_input[idx])
        ax.get_xaxis().set_visible(False)
        

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        if i == 0:
            ax.set_ylabel("Output", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_decoded[idx])
        ax.get_xaxis().set_visible(False)

num_time_series = 100

input_path = os.path.join(os.path.abspath(__file__), "../../../dir/")
input_path = os.path.join(input_path, input_file)

query_path = os.path.join(os.path.abspath(__file__), "../../../dir/")
query_path = os.path.join(query_path, query_file)

encoder_path = os.path.join(os.path.abspath(__file__), "../../../models/encoder.h5")
autoencoder_path = os.path.join(os.path.abspath(__file__), "../../../models/autoencoder.h5")

df_input = pd.read_csv(input_path, header=None, delimiter='\t')
file_ids_input = df_input.iloc[:, [0]].values
df_input = df_input.drop(df_input.columns[0], axis=1)
df_input = df_input.transpose()

df_query = pd.read_csv(query_path, header=None, delimiter='\t')
file_ids_query = df_query.iloc[:, [0]].values
df_query = df_query.drop(df_query.columns[0], axis=1)
df_query = df_query.transpose()

# Training data: 90%, Test data: 10%
train_size = int(len(df_input) * 0.90)
test_size = len(df_input) - train_size
print("Train size: ", train_size)

scaler = MinMaxScaler()

x_train = []
for step in range (0, num_time_series-10):
    
    training_set = df_input.iloc[:train_size, [step]].values
    training_set_scaled = scaler.fit_transform(training_set)

    for i in range(window_length, train_size):
        x_train.append(training_set_scaled[i-window_length:i, 0])

x_train = np.array(x_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

x_train = x_train.astype('float32')

# Load model
autoencoder = load_model(autoencoder_path)
encoder = load_model(encoder_path)

for step in range (0, num_time_series-10):
    
    dataset_train = df_input.iloc[:train_size, [step]]
    dataset_test = df_input.iloc[train_size:, [step]]
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)

    inputs = dataset_total[len(dataset_total) - len(dataset_test) - window_length:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)
    x_test = []
    for i in range(window_length, test_size+window_length):
        x_test.append(inputs[i-window_length:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    x_test = x_test.astype('float32')

    time_series = []

    data = df_input.iloc[0:, [step]].values
    data_scaled = scaler.fit_transform(data)

    for i in range(window_length, len(data), window_length):
        time_series.append(data_scaled[i-window_length:i, 0])

    time_series = np.array(time_series)
    time_series = np.reshape(time_series, (time_series.shape[0], time_series.shape[1], 1))
    time_series = time_series.astype('float32')

    # decoded_stocks = autoencoder.predict(x_test)

    encoder.compile(optimizer='adam', loss='binary_crossentropy')
    encoded_stocks = encoder.predict(time_series)

    encoded_stocks = np.reshape(encoded_stocks, (encoded_stocks.shape[0]*3, 1))

    encoded_val= scaler.inverse_transform(encoded_stocks)

    with open(os.path.join(os.path.abspath(__file__), "../../../dir/temp.csv"),'a') as a_file:
        a_file.write(str(file_ids_input[step]))
        a_file.write('\t')
        for i in range(0, len(encoded_val)):
            a_file.write(str(encoded_val[i]))
            a_file.write('\t')
        a_file.write("\n")
        a_file.close()
        

    with open(os.path.join(os.path.abspath(__file__), "../../../dir/temp.csv"), 'r') as infile, open(os.path.join(os.path.abspath(__file__), "../../../dir/") + out_input_file, 'w') as outfile:
        data = infile.read()
        data = data.replace("[", "")
        data = data.replace("]", "")
        data = data.replace("'", "")
        outfile.write(data)
        infile.close()
        outfile.close()

    # plot_examples(x_test, decoded_stocks)

os.remove(os.path.join(os.path.abspath(__file__), "../../../dir/temp.csv"))

for step in range (0, 10):
    
    dataset_train = df_query.iloc[:train_size, [step]]
    dataset_test = df_query.iloc[train_size:, [step]]
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)

    inputs = dataset_total[len(dataset_total) - len(dataset_test) - window_length:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)
    x_test = []
    for i in range(window_length, test_size+window_length):
        x_test.append(inputs[i-window_length:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    x_test = x_test.astype('float32')

    time_series = []

    data = df_query.iloc[0:, [step]].values
    data_scaled = scaler.fit_transform(data)

    for i in range(window_length, len(data), window_length):
        time_series.append(data_scaled[i-window_length:i, 0])

    time_series = np.array(time_series)
    time_series = np.reshape(time_series, (time_series.shape[0], time_series.shape[1], 1))
    time_series = time_series.astype('float32')

    # decoded_stocks = autoencoder.predict(x_test)

    encoder.compile(optimizer='adam', loss='binary_crossentropy')
    encoded_stocks = encoder.predict(time_series)

    encoded_stocks = np.reshape(encoded_stocks, (encoded_stocks.shape[0]*3, 1))

    encoded_val= scaler.inverse_transform(encoded_stocks)

    with open(os.path.join(os.path.abspath(__file__), "../../../dir/temp.csv"),'a') as a_file:
        a_file.write(str(file_ids_query[step]))
        a_file.write('\t')
        for i in range(0, len(encoded_val)):
            a_file.write(str(encoded_val[i]))
            a_file.write('\t')
        a_file.write("\n")
        a_file.close()
        

    with open(os.path.join(os.path.abspath(__file__), "../../../dir/temp.csv"), 'r') as infile, open(os.path.join(os.path.abspath(__file__), "../../../dir/") + out_query_file, 'w') as outfile:
        data = infile.read()
        data = data.replace("[", "")
        data = data.replace("]", "")
        data = data.replace("'", "")
        outfile.write(data)
        infile.close()
        outfile.close()

    # plot_examples(x_test, decoded_stocks)

os.remove(os.path.join(os.path.abspath(__file__), "../../../dir/temp.csv"))