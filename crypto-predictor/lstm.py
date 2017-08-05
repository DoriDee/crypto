
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Dense, Activation, Dropout

def build_model2(layers):
    model = Sequential()

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    # start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    # print("> Compilation Time : ", time.time() - start)
    return model

def build_model(seq_len, feature_num=1):
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(seq_len, feature_num)))
    # model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model