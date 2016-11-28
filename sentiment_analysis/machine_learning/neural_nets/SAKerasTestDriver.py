# CNN for the IMDB problem
import pickle

import math
import numpy
import sklearn
import theano
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.utils import np_utils

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import KFold, cross_val_score

#load dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

train_data = numpy.load("C:/Users/user/PycharmProjects/ms-thesis/word_embeddings/vanzo_train.npz")
test_data = numpy.load("C:/Users/user/PycharmProjects/ms-thesis/word_embeddings/vanzo_test.npz")

X_train = train_data["X"]
Y_train = train_data["Y"]

X_test = test_data["X"]
Y_test = test_data["Y"]


# Scale data if needed
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# encoder = LabelEncoder()
# encoder.fit(Y_train)
# Y_train = encoder.transform(Y_train)
# Y_test = encoder.transform(Y_test)
# # convert integers to dummy variables (i.e. one hot encoded)
Y_train_ohe = np_utils.to_categorical(Y_train)
Y_test_ohe = np_utils.to_categorical(Y_test)



# from itertools import groupby
# for key, group in groupby(Y_test):
#     print("{}-{}".format(key, len(list(group))))

# create the model
def build_model():

    model = Sequential()

    model.add(Dense(300, input_dim=300, init='uniform', activation='tanh'))
    model.add(Dense(300, init='uniform', activation='tanh'))
    model.add(Dense(3, init="uniform", activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # model.add(Convolution1D(nb_filter=32, filter_length=3, activation="tanh", border_mode="same", input_shape=(300,)))
    # model.add(MaxPooling1D(pool_length=2))
    # model.add(Flatten())
    # model.add(Dense(250, activation="relu"))
    # model.add(Dense(1, activation="sigmoid"))
    # model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    print(model.summary())
    return model

# Fit the model
model = build_model()
model.fit(X_train, Y_train_ohe, validation_data=(X_test, Y_test_ohe), nb_epoch=100, batch_size=128, verbose=1)

print("Train: {}, Test: {}".format(len(X_train), len(X_test)))

# Final evaluation of the model
scores = model.evaluate(X_test, Y_test_ohe, verbose=2)
print("Accuracy: %.2f%%" % (scores[1]*100))



predicted_arr = model.predict_classes(X_test)
actual_arr = Y_test

print('Accuracy: {}\n'.format(sklearn.metrics.accuracy_score(actual_arr, predicted_arr)))
print(sklearn.metrics.classification_report(actual_arr, predicted_arr))
print(numpy.array_str(sklearn.metrics.confusion_matrix(actual_arr, predicted_arr))) # ordering is alphabetical order of label names


# from keras.utils.visualize_util import plot
# plot(model, to_file='model.png', show_shapes=True)


# 10-fold cross validation
def kfold_validation(X, Y, n_folds):
    estimator = KerasClassifier(build_fn=build_model, nb_epoch=200, batch_size=128, verbose=2)
    k_fold = KFold(n=n_folds, n_folds=n_folds, shuffle=False, random_state=None)
    results = cross_val_score(estimator, X, Y, cv=k_fold)
    print(results)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# kfold_validation(X_test, Y_test, 10)