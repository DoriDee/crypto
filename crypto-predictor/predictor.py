# LSTM for international airline passengers problem with time step regression framing
import numpy
from pandas import read_csv
import math
from sklearn.preprocessing import MinMaxScaler

from collections import deque
import csv
from logger import Logger

def get_last_row(csv_filename):
    with open(csv_filename, 'rU') as f:
        try:
            lastrow = deque(csv.reader(f), 1)[0]
        except IndexError:  # empty file
            lastrow = None
        return lastrow

def load_dataset(file_name):
    # load the dataset
    dataframe = read_csv(file_name, 
                         usecols=[2],
                         engine='python',
                         skipfooter=False,
                         keep_default_na=False,
                         skip_blank_lines=True)
    
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
    
    Logger.log_writer("predictor#getting last row!")

    last_row = get_last_row(file_name)

    Logger.log_writer("predictor#got last row!")
    
    last_value = last_row[2]
    market_cap = last_row[-1]

    Logger.log_writer("last_value:{0}".format(last_value))
    Logger.log_writer("market_cap:{0}".format(market_cap))

    print("last_value:{0}".format(last_value))
    print("market_cap:{0}".format(market_cap))
    
    seq_len = 50
    
    Logger.log_writer("predictor#Loading dataset..")

    dataset = load_dataset(file_name)

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    feature_num = dataset.shape[1]

    Logger.log_writer("predictor#Preparing data..")
    
    trainX, trainY = prepare_data(dataset, seq_len)

    import lstm

    Logger.log_writer("predictor#Building model..")
    
    model = lstm.build_model2([1, 50, 100, 1])
    # model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)
    model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)

    Logger.log_writer("predictor#predicting..")
    
    trainPredict = model.predict(trainX)    
    trainPredict = scaler.inverse_transform(trainPredict)

    Logger.log_writer("predictor#predicted!")
    print("predictor#coocoooo!")
    print(trainPredict[-1,-1])

    Logger.log_writer("predictor#done! lastValue:{0}, pred: {1}, cap: {2}".format(last_value, trainPredict[-1,-1], market_cap))
    
    return last_value, trainPredict[-1,-1], market_cap

