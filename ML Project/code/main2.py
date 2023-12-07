import cv2 as cv
import numpy as np
import pandas as pd
import random
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split

data = pd.read_csv("../preprocessed/n1000__shuffle1_seed123/all.csv")
# print(len(data))

X = []
Y = []
for i in range(len(data)):
    img = cv.imread(data.iloc()[i].filename, 0)
    img = cv.resize(img, (360, 320))
    X.append(img)
    Y.append(data.iloc()[i].label)

num_classes = 3
X = np.array(X)/255
Y = np.array(Y)
print(pd.Series(Y).value_counts())
X = np.expand_dims(X, -1)
# Y = keras.utils.to_categorical(Y, num_classes)
print(X.shape)
print(Y.shape)

test_percs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

for test_size in test_percs:

    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=test_size)
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)

    # x_train_0 = []
    # x_train_1 = []
    # y_train_0 = []
    # y_train_1 = []
    # for i in range(len(x_train)):
    #     if y_train[i] == 2:
    #         x_train_1.append(x_train[i])
    #         y_train_1.append(y_train[i])
    #     else:
    #         x_train_0.append(x_train[i])
    #         y_train_0.append(y_train[i])
    # x_train_0 = np.array(x_train_0)
    # x_train_1 = np.array(x_train_1)
    # y_train_0 = np.array(y_train_0)
    # y_train_1 = np.array(y_train_1)
    # print("______________________")
    # print(x_train_0.shape)
    # print(x_train_1.shape)
    # print(y_train_0.shape)
    # print(y_train_1.shape)
    # print("______________________")

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    # y_train_0 = keras.utils.to_categorical(y_train_0, num_classes)
    # y_train_1 = keras.utils.to_categorical(y_train_1, num_classes)

    input_shape = (1248, 320, 360, 1)
    cnn_model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape[1:], activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(rate=0.5),
        Dense(units=num_classes, activation='softmax')
    ])
    # print(cnn_model.summary())

    cnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    batch_size = 64
    epochs = 2

    cnn_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    score = cnn_model.evaluate(x_test, y_test, verbose=0)
    print(f"for test size:{test_size}, accuracy is:{score[1]}")
