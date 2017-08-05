import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from datetime import timedelta
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
import random

from pathlib import Path

def convert_btc_to_usd(coin_name, normalized=0):
    col_names = ['Date','Shit', 'last', 'high', 'low', 'volume']
    btc_data = pd.read_csv('./history/btc_usd.csv', header=0)
    
    file_name = './history/full/bittrex_{}_BTC1day.csv'.format(coin_name)

    my_file = Path(file_name)
    if not my_file.is_file():
        # print("File " + file_name + " doesn't exist")
        return

    print(coin_name)

    coin_data = pd.read_csv(file_name, header=None, names=col_names) 
    
    df_coin = pd.DataFrame(coin_data)
    df_btc = pd.DataFrame(btc_data)

    usd_prices = []

    for index in range(len(coin_data)):    
        btc_price = df_btc['Close Price'][index]
        coin_price = float(df_coin['last'][index - 1]) if df_coin['last'][index].strip() == '' else float(df_coin['last'][index])
        price_in_usd = coin_price * btc_price
        usd_prices.append(price_in_usd)

    # print(df['Date'])

    # print("======")
    # print(df['Date'].str)
    # print("======")
    # print(df['Date'].str.split('-'))

    # date_split = df['Date'].str.split('-').str

    # print(date_split)

    # df['Year'], df['Month'], df['Day'] = date_split
    # df["Volume"] = df["Volume"] / 10000
    # df.drop(df.columns[[0,3,5,6, 7,8,9]], axis=1, inplace=True) 
    return "df"


top_coins = ['BTC', 'ETH', 'XRP', 'CUBE', 'LTC', 'ETC', 'XEM', 'DASH', 'IOT', 'XMR', 'BTS', 'STRAT', 'ZEC',
             'STEEM', 'WAVES', 'ANS', 'GNT', 'BCN', 'BCC', 'SC', 'GNO', 'VERI', 'ICN', 'REP', 'LSK', 'DOGE',
             'XLM', 'USDT', 'GBYTE', 'MAID', 'FCT']

for index in range(len(top_coins)):
    convert_btc_to_usd(top_coins[index])

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def normalise_windows(window_data):
    normalised_data = []

    for coin_index in range(len(window_data[0][0])):
        for window in window_data:
            p_0 = window[0][coin_index]
            
            for seq_index in (range(len(window))):
                window[seq_index][coin_index] = (window[seq_index][coin_index] / p_0) - 1
            
    return window_data

def load_dummy_data(seq_len, target_coin_name = 'uppy'): 

    coin_names = ['btc', 'downy']
    col_names = ['Date', 'Price']
    
    sequence_length = seq_len + 1
    coin_index = 0 

    coins_raw_data = []
    coins_lengths = []

    file_name = "dummy_{}.csv".format(target_coin_name)
    target_coin_data = pd.read_csv(file_name, header=0, names=col_names)
    start_date = target_coin_data['Date'][0]
    end_date = target_coin_data['Date'].iloc[-1]

    for coin_name in coin_names:
        file_name = "dummy_{}.csv".format(coin_name)
        coin_data = pd.read_csv(file_name, header=0, names=col_names)

        # coin_data['Date'] = pd.to_datetime(coin_data['Date'])  

        # Pad data frame
        if coin_data['Date'][0] > start_date:
            dates_pad = pd.date_range(start_date, datetime.datetime.strptime(coin_data['Date'][0], "%Y-%m-%d") - timedelta(days=1), freq='D')
            df_pad = pd.DataFrame({ 'Price': coin_data['Price'][0], 'Date':dates_pad })
            coin_prices = list(df_pad.append(coin_data)['Price'])
        elif coin_data['Date'][0] < start_date:
            start_index = list(coin_data['Date']).index(start_date)
            end_index = list(coin_data['Date']).index(end_date)
            coin_prices = list(coin_data['Price'])[start_index:end_index + 1]
            
        coins_raw_data.append(coin_prices)
    
    coins_raw_data.append(list(target_coin_data['Price']))
    
    # for x in range(len(coins_raw_data)):
    #     plt.plot(coins_raw_data[x], label=coin_name)
    #     plt.xlabel('date')
    #     plt.ylabel('price')
    # plt.legend(loc='upper right')
    # plt.show()
    
    # set the coins as a matrix and normalize each one of the cols
    matrix = [[0 for x in range(len(coins_raw_data))] for y in range(len(coins_raw_data[0]))]

    for x in range(len(coins_raw_data[0])):
        for y in range(len(coins_raw_data)):
            matrix[x][y] = coins_raw_data[y][x]

    windows = []

    for win_index in range(len(coins_raw_data[0]) - sequence_length):
        win = [[0 for x in range(len(coins_raw_data))] for y in range(sequence_length)]
        for seq_index in range(sequence_length):
            for coin_index in range(len(coins_raw_data)):        
                win[seq_index][coin_index] = matrix[win_index + seq_index][coin_index] #.a.astype(float).fillna(0.0)
        windows.append(win)

    normalized_windows = normalise_windows(windows)

    result = np.array(normalized_windows)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], len(matrix[0])))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], len(matrix[0])))  

    return [x_train, y_train, x_test, y_test]


def create_dummy_sets(start_date, coin_name):

    df_dummy_btc = {'date' : [], 'price' : []}

    end_date = datetime.date( year = 2017, month = 6, day = 30 )
    d = start_date
    delta = datetime.timedelta(days=1)
    price = 0.01

    if (coin_name == 'btc'):
        factor = 2
    elif (coin_name == 'uppy'):
        factor = 3
    elif (coin_name == 'downy'):
        factor = -2
    
    change_periods = [{'start': datetime.date( year = 2015, month = 2, day = 20 ),
                       'end': datetime.date( year = 2015, month = 4, day = 20 )},
                       {'start': datetime.date( year = 2016, month = 2, day = 20 ),
                       'end': datetime.date( year = 2016, month = 4, day = 20 )},
                       {'start': datetime.date( year = 2017, month = 4, day = 20 ),
                       'end': datetime.date( year = 2017, month = 5, day = 20 )},
                       {'start': datetime.date( year = 2017, month = 6, day = 20 ),
                       'end': datetime.date( year = 2017, month = 7, day = 20 )}]

    while d <= end_date:
        date = d.strftime("%Y-%m-%d") 
        # print(date)
        d += delta    
        df_dummy_btc['date'].append(d)
        df_dummy_btc['price'].append(price) 

        price_delta = 0.5 + random.uniform(-1, 1)
        is_change_period = False
        for n in change_periods:
            if n['start'] <= d <= n['end']:
                is_change_period = True

        if is_change_period:
            price_delta += factor * random.random()

        price += price_delta

    df = pd.DataFrame(df_dummy_btc)    

    df.to_csv('dummy_{}.csv'.format(coin_name), index=False)


    plt.plot(df['date'], df['price'], label=coin_name)
    plt.xlabel('date')
    plt.ylabel('price')

# create_dummy_sets(datetime.date( year = 2014, month = 1, day = 1 ), 'btc')
# create_dummy_sets(datetime.date( year = 2015, month = 1, day = 1 ), 'uppy')
# create_dummy_sets(datetime.date( year = 2015, month = 6, day = 1 ), 'downy')

# plt.legend(loc='upper right')
# plt.show()

seq_len = 50

X_train, y_train, X_test, y_test = load_dummy_data(seq_len)

import lstm

model = lstm.build_model([3, seq_len, 100, 1])

model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=50,
    validation_split=0.05)

# predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
# predicted = lstm.predict_sequence_full(model, X_test, seq_len)
predicted = lstm.predict_point_by_point(model, X_test)        

# print('Training duration (s) : ', time.time() - global_start_time)

# trainScore = model.evaluate(X_train, y_train, verbose=0)
# print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

# testScore = model.evaluate(X_test, y_test, verbose=0)
# print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

# plot_results_multiple(predictions, y_test, 50)
plot_results(predicted,  y_test)



def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()
