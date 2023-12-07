import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv("../preprocessed/n1000__shuffle1_seed123/all.csv")
# print(len(data))

X = []
Y = []
for i in range(len(data)):
    img = cv.imread(data.iloc()[i].filename, 0)
    img = cv.resize(img, (16, 16))
    X.append(img)
    if data.iloc()[i].label == 2:
        Y.append(1)
    else:
        Y.append(0)

X = np.array(X)
Y = np.array(Y)
# print(pd.Series(Y).value_counts())
# print(X)
# print(X.shape)

X = X.reshape(len(X), -1)
# print(X.shape)

test_percs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

for test_size in test_percs:
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=test_size)
    # print(x_train)
    # print(x_train.shape)
    # print(x_test.shape)
    # print(y_train.shape)

    # 3.1: Extraction of feature descriptors
    sift = cv.SIFT.create()
    kp_train, FD_train = sift.detectAndCompute(x_train, None)
    kp_test, FD_test = sift.detectAndCompute(x_test, None)
    FD_train = np.array(FD_train)
    FD_test = np.array(FD_test)
    # FD_train = np.tile(FD_train, (1, 2))
    # FD_test = np.tile(FD_test, (1, 2))

    # 3.2: Clustering of feature descriptors
    kmeans = KMeans(n_clusters=500, init="k-means++", n_init="auto")
    kmeans_train = kmeans.fit(FD_train)
    kmeans_test = kmeans.fit(FD_test)
    VV_train = kmeans_train.cluster_centers_
    VV_test = kmeans_test.cluster_centers_
    VV_train = np.array(VV_train)
    VV_test = np.array(VV_test)
    VV_train = np.tile(VV_train, (1, 2))
    VV_test = np.tile(VV_test, (1, 2))
    # print(VV_train)
    # print(VV_train.shape)

    # 3.3:  Classification of images
    train_distances = np.column_stack([np.sum((x_train - center) ** 2, axis=1) ** 0.5 for center in VV_train])
    train_distances = np.array(train_distances)
    # print(distances)
    # print(distances.shape)
    test_distances = np.column_stack([np.sum((x_test - center) ** 2, axis=1) ** 0.5 for center in VV_test])

    svm = SVC()
    svm.fit(train_distances, y_train)

    y_prd = svm.predict(test_distances)
    print(f"for test size:{test_size}, accuracy:{accuracy_score(y_test, y_prd)}")
    cfm = confusion_matrix(y_test, y_prd)
    print(cfm)
