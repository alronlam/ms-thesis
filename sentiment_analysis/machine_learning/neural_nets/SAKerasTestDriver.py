# CNN for the IMDB problem
import pickle

import math
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

import random
from random import shuffle

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import KFold, cross_val_score

from twitter_data.parsing.folders import FolderIO
from twitter_data.database import DBManager
from sentiment_analysis.evaluation import TSVParser
# fix random seed for reproducibility
# seed = 7
# numpy.random.seed(seed)

# load the dataset but only keep the top n words, zero the rest
# top_words = 5000
# test_split = 0.33


#load dataset
train_data = numpy.load("C:/Users/user/PycharmProjects/ms-thesis/word_embeddings/vanzo_train.npz")
test_data = numpy.load("C:/Users/user/PycharmProjects/ms-thesis/word_embeddings/vanzo_test.npz")

X_train = train_data["X"]
Y_train = train_data["Y"]

X_test = test_data["X"]
Y_test = test_data["Y"]

# X = numpy.concatenate(X_train, X_test)
# Y = numpy.concatenate(Y_train, Y_test)

# create the model
def build_model():
    model = Sequential()

    model.add(Dense(300, input_dim=300, init='normal', activation='relu'))
    model.add(Dense(3, init="normal", activation="sigmoid"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # model.add(Convolution1D(nb_filter=32, filter_length=3, activation="tanh", border_mode="same", input_shape=(300,)))
    # model.add(MaxPooling1D(pool_length=2))
    # model.add(Flatten())
    # model.add(Dense(250, activation="relu"))
    # model.add(Dense(1, activation="sigmoid"))
    # model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # print(model.summary())
    return model

# 10-fold cross validation
def kfold_validation(X, Y, n_folds):
    estimator = KerasClassifier(build_fn=build_model, nb_epoch=200, batch_size=128, verbose=1)
    k_fold = KFold(n=n_folds, n_folds=n_folds, shuffle=True, random_state=None)
    results = cross_val_score(estimator, X, Y, cv=k_fold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Fit the model
model = build_model()
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=200, batch_size=128, verbose=1)

print("Train: {}, Test: {}".format(len(X_train), len(X_test)))

# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))