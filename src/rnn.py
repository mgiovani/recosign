import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
import numpy
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

from data import x_treino, x_teste, y_treino, y_teste 

x_treino = minmax_scale(x_treino)
x_teste = minmax_scale(x_teste)
x_treino = x_treino.reshape(150, 300, -1)
x_teste = x_teste.reshape(40, 300, -1)


def rnn_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(300, 1), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='softmax'))
    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-4)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )
    return model


model = rnn_model()
model.fit(x_treino, y_treino, epochs=10, validation_data=(x_teste, y_teste))
