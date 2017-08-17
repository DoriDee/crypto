# LSTM for international airline passengers problem with time step regression framing
import numpy
from pandas import read_csv
import math
from sklearn.preprocessing import MinMaxScaler

from collections import deque
import csv

def get_last_row(csv_filename):
    with open(csv_filename, 'r') as f:
        try:
            lastrow = deque(csv.reader(f), 1)[0]
        except IndexError:  # empty file
            lastrow = None
        return lastrow

def load_dataset(file_name):
    # load the dataset
    dataframe = read_csv(file_name, usecols=[2], engine='python', skipfooter=False, keep_default_na=False)
    
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    return dataset

def prepare_data(dataset, seq_len):
    
    # split into train and test sets
    train = dataset[0:len(dataset),:]

    # reshape into X=t and Y=t+1    
    trainX, trainY = create_dataset(train, seq_len)
    
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    
    return [trainX, trainY]

# convert an array of values into a dataset matrix
def create_dataset(dataset, seq_len=1, predict_next = False):
	dataX, dataY = [], []
	limit = len(dataset)-seq_len + 1 if predict_next else len(dataset)-seq_len

	for i in range(limit):
		a = dataset[i:(i+seq_len), 0]
		dataX.append(a)
		if (i + seq_len) < len(dataset):
		    dataY.append(dataset[i + seq_len, 0])
	return numpy.array(dataX), numpy.array(dataY)

def predict(file_name):
    
    last_row = get_last_row(file_name)

    last_value = last_row[2]
    market_cap = last_row[-1]

    print("last_value:{0}".format(last_value))
    print("market_cap:{0}".format(market_cap))
    
    seq_len = 50
    
    dataset = load_dataset(file_name)

    # last_value = dataset[-1]

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    feature_num = dataset.shape[1]

    trainX, trainY = prepare_data(dataset, seq_len)

    import lstm

    model = lstm.build_model2([1, 50, 100, 1])
    # model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)
    model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)

    trainPredict = model.predict(trainX)    
    trainPredict = scaler.inverse_transform(trainPredict)

    print(trainPredict)
    print("coocoooo!")
    print(trainPredict[-1,-1])

    return last_value, trainPredict[-1,-1], market_cap


predict('LTC.csv')