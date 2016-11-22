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

from random import shuffle

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import KFold, cross_val_score

from twitter_data.parsing.folders import FolderIO
from twitter_data.database import DBManager
from sentiment_analysis.evaluation import TSVParser
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
test_split = 0.33

# create the model



#load dataset
pickle_file = open("C:/Users/user/PycharmProjects/ms-thesis/word_embeddings/vanzo_dataset_google_embeddings.pickle", "rb")
# (X,Y) = pickle.load(pickle_file)
data = numpy.load("C:/Users/user/PycharmProjects/ms-thesis/word_embeddings/vanzo_dataset.npz")
X = data['X']
Y = data['Y']

shuffle(X)
shuffle(Y)
boundary = math.floor(len(X) * 0.7)

def build_model():
    model = Sequential()

    model.add(Dense(12, input_dim=300, init='normal', activation='relu'))
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
estimator = KerasClassifier(build_fn=build_model, nb_epoch=200, batch_size=128, verbose=1)
k_fold = KFold(5, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=k_fold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# Partition the dataset
# shuffle(X)
# shuffle(Y)
# boundary = math.floor(len(X) * 0.7)
#
# X_train = X[:boundary]
# X_test = X[boundary:]
# Y_train = Y[:boundary]
# Y_test = Y[boundary:]

# Fit the model
# model = build_model()
# model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=2, batch_size=128, verbose=1)
# # Final evaluation of the model
# scores = model.evaluate(X_test, Y_test, verbose=1)
# print("Accuracy: %.2f%%" % (scores[1]*100))