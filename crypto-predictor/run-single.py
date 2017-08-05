# LSTM for international airline passengers problem with time step regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def load_dataset(file_name):
    # load the dataset
    dataframe = read_csv(file_name, usecols=[2], engine='python', skipfooter=False)

    # dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    return dataset

def load_dataset_multi(file_name):
    # load the dataset
    dataframe = read_csv(file_name, usecols=[2, 5], engine='python', skipfooter=False)
    array_len = dataframe.shape[0]
    val_len = dataframe.shape[1] 
    matrix = [[0 for x in range(val_len)] for y in range(array_len)]
    
    dataset = dataframe.values

    for i in range(array_len):
        for y in range(val_len):
            val = dataset[i][y]
            if val.strip() == '':
                val = dataset[i-1][y]        
            matrix[i][y] = float(val)
    
    return matrix

def prepare_data(dataset, seq_len):
    
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    # reshape into X=t and Y=t+1    
    trainX, trainY = create_dataset(train, seq_len)
    testX, testY = create_dataset(test, seq_len, True)

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    return [trainX, trainY, testX, testY]

def prepare_data_multi(dataset, seq_len, feature_num):
    
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    # reshape into X=t and Y=t+1    
    trainX, trainY = create_dataset_multi(train, seq_len)
    testX, testY = create_dataset_multi(test, seq_len)

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], feature_num))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], feature_num))

    # scaleX.fit_transform(trainX)
    # scaleY.fit_transform(trainY)

    return [trainX, trainY, testX, testY]

def predict(model, trainX, testX):
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    return [trainPredict, testPredict]
    
def denormalize(trainPredict, testPredict, trainY, testY, dataset):
    
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)

    trainY = scaler.inverse_transform([trainY])
    testY = scaler.inverse_transform([testY])
    dataset = scaler.inverse_transform(dataset)

    return [trainPredict, testPredict, trainY, testY, dataset]

def denormalize_multi(trainPredict, testPredict, trainY, testY):
    # invert predictions
    # trainPredict = trainPredict.reshape(len(trainPredict),)
    # testPredict = testPredict.reshape(len(testPredict),)

    trainPredict = scaleY.inverse_transform(trainPredict)
    testPredict = scaleY.inverse_transform(testPredict)

    trainY = scaleY.inverse_transform([trainY])
    testY = scaleY.inverse_transform([testY])

    return [trainPredict, testPredict, trainY, testY]

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

def create_dataset_multi(dataset, seq_len = 1):
    dataX, dataY = [], []
    for i in range(len(dataset)-seq_len-1):
        a = dataset[i:(i+seq_len)]
        dataX.append(a)
        dataY.append(dataset[i + seq_len, 0])
    return numpy.array(dataX), numpy.array(dataY)

seq_len = 50

# fix random seed for reproducibility
numpy.random.seed(7)

dataset = load_dataset('bittrex_LTC_BTC1day.csv')
# dataset = load_dataset('btc_usd1day.csv')
# dataset = load_dataset('realbtc.csv')
# dataset = load_dataset('ltc_usd1day.csv')

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

feature_num = dataset.shape[1]

trainX, trainY, testX, testY = prepare_data(dataset, seq_len)

import lstm

# model = lstm.build_model(seq_len, feature_num)
# Model2 is better!!
model = lstm.build_model2([1, 50, 100, 1])
model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)
# model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
# model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

trainPredict, testPredict = predict(model, trainX, testX)

trainPredict, testPredict, trainY, testY, dataset = denormalize(trainPredict, testPredict, trainY, testY, dataset)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:-1,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[seq_len:len(trainPredict)+seq_len, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(numpy.append(dataset, 0).reshape(len(dataset) + 1,1))
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(seq_len*2):len(dataset) + 1, :] = testPredict
# testPredictPlot[len(trainPredict)+(seq_len*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()