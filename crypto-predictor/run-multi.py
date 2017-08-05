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
    # dataframe = read_csv(file_name, usecols=[2, 5], engine='python', skipfooter=False)
    dataframe = read_csv(file_name, usecols=[3, 6], engine='python', skipfooter=False)
    array_len = dataframe.shape[0]
    val_len = dataframe.shape[1] 
    matrix = [[0 for x in range(val_len)] for y in range(array_len)]
    
    dataset = dataframe.values

    for i in range(array_len):
        for y in range(val_len):
            val = dataset[i][y]
            # if val.strip() == '':
            #     val = dataset[i-1][y]        
            matrix[i][y] = float(val)
    
    return matrix

def prepare_data(dataset, seq_len):
    
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    # reshape into X=t and Y=t+1    
    trainX, trainY = create_dataset(train, seq_len)
    testX, testY = create_dataset(test, seq_len)

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

    # scaleX.fit_transform(trainX)
    # scaleY.fit_transform(trainY)

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], feature_num))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], feature_num))

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

    # create empty table with 2 fields
    trainPredict_dataset_like = numpy.zeros(shape=(len(trainPredict), 2) )
    # put the predicted values in the right field
    trainPredict_dataset_like[:,0] = trainPredict[:,0]
    # inverse transform and then select the right field
    trainPredict = scalerDS.inverse_transform(trainPredict_dataset_like)[:,0]

    # create empty table with 2 fields
    testPredict_dataset_like = numpy.zeros(shape=(len(testPredict), 2) )
    # put the predicted values in the right field
    testPredict_dataset_like[:,0] = testPredict[:,0]
    # inverse transform and then select the right field
    testPredict = scalerDS.inverse_transform(testPredict_dataset_like)[:,0]
    
    # trainPredict = scalerDS.inverse_transform(trainPredict)
    # testPredict = scalerDS.inverse_transform(testPredict)

    # trainY = scalerDS.inverse_transform([trainY])
    # testY = scalerDS.inverse_transform([testY])

    trainY_dataset_like = numpy.zeros(shape=(len(trainY), 2) )
    trainY_dataset_like[:,0] = trainY[:]
    trainY = scalerDS.inverse_transform(trainY_dataset_like)[:,0]

    testY_dataset_like = numpy.zeros(shape=(len(testY), 2) )
    testY_dataset_like[:,0] = testY[:]
    testY = scalerDS.inverse_transform(testY_dataset_like)[:,0]

    return [trainPredict, testPredict, trainY, testY]

# convert an array of values into a dataset matrix
def create_dataset(dataset, seq_len=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-seq_len-1):
		a = dataset[i:(i+seq_len), 0]
		dataX.append(a)
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

# dataset = load_dataset('bittrex_LTC_BTC1day.csv')
# dataset = load_dataset_multi('bittrex_LTC_BTC1day.csv')
dataset = load_dataset_multi('btc_usd1day.csv')
# dataset = load_dataset_multi('ltc_usd1day.csv')
# dataset = numpy.array(dataset)
# dataset = normalize(dataset)

scalerDS = MinMaxScaler(feature_range=(0, 1))
dataset = scalerDS.fit_transform(dataset)
scaleX = MinMaxScaler(feature_range=(0, 1))
scaleY = MinMaxScaler(feature_range=(0, 1))

feature_num = dataset.shape[1]

trainX, trainY, testX, testY = prepare_data_multi(dataset, seq_len, feature_num)
# trainX, trainY, testX, testY = prepare_data(dataset, seq_len)

import lstm

model = lstm.build_model(seq_len, feature_num)
model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)
# model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

trainPredict, testPredict = predict(model, trainX, testX)

# trainPredict, testPredict, trainY, testY, dataset = denormalize(trainPredict, testPredict, trainY, testY, dataset)
trainPredict, testPredict, trainY, testY = denormalize_multi(trainPredict, testPredict, trainY, testY)

# calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset[:,0])
trainPredictPlot[:] = numpy.nan
trainPredictPlot[seq_len:len(trainPredict)+seq_len] = trainPredict[:]

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset[:, 0])
testPredictPlot[:] = numpy.nan
testPredictPlot[len(trainPredict)+(seq_len*2)+1:len(dataset)-1] = testPredict[:]

# plot baseline and predictions
# plt.plot(dataset[0:-1:,0])
plt.plot(scalerDS.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()