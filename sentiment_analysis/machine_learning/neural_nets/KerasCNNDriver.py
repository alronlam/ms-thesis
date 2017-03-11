import os
import pickle
from itertools import groupby

from keras.engine import Model, Input, merge
from keras.models import load_model, model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

import numpy as np



#####################
##### Constants #####
#####################

TEXT_DATA_DIR = "20_newsgroup"
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 32
MAX_CONTEXTUAL_WORDS = 32
VALIDATION_SPLIT = 0.1
GLOVE_DIR = "glove"
GLOVE_FILE = "glove.twitter.27B.200d.txt"
EMBEDDING_DIM = 200

##########################################
##### Functions for loading datasets #####
##########################################
def load_vanzo_dataset():
    data = np.load("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/feature_extraction/word_embeddings/vanzo_word_sequence_concat_glove_200d_preprocessed.npz")
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_conv_train = data["x_conv_train"]

    x_test = data["x_test"]
    y_test = data["y_test"]
    x_conv_test = data["x_conv_test"]

    embedding_matrix = data["embedding_matrix"]

    return (x_train, y_train, x_conv_train, x_test, y_test, x_conv_test, embedding_matrix)

#########################################################################################################

##############################################
### Create the Neural Network Architecture ###
##############################################

from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, AveragePooling1D, Lambda

######################
### Custom Pooling ###
######################

from keras import backend as K

def max_min_avg_pooling_main(x):
    MAX_SEQUENCE_LENGTH = 32
    from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, AveragePooling1D, Lambda
    max_arr = MaxPooling1D(MAX_SEQUENCE_LENGTH-2)(x)
    # TODO not sure why the pool size and strides work, but verified manually (through inspecting input and ouput that it produces correct output)
    min_arr = -K.pool2d(-x, pool_size=(MAX_SEQUENCE_LENGTH-2,1), strides=(MAX_SEQUENCE_LENGTH-2,1),
                          border_mode="valid", dim_ordering="th", pool_mode='max')
    avg_arr = AveragePooling1D(MAX_SEQUENCE_LENGTH-2)(x)

    return K.concatenate([max_arr, min_arr, avg_arr], axis=1)


def max_min_avg_pooling_context(x):
    MAX_CONTEXTUAL_WORDS = 32
    from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, AveragePooling1D, Lambda
    max_arr = MaxPooling1D(MAX_CONTEXTUAL_WORDS)(x)
    # TODO not sure why the pool size and strides work, but verified manually (through inspecting input and ouput that it produces correct output)
    min_arr = -K.pool2d(-x, pool_size=(32,1), strides=(32,1),
                          border_mode="valid", dim_ordering="th", pool_mode='max')
    avg_arr = AveragePooling1D(MAX_CONTEXTUAL_WORDS)(x)

    return K.concatenate([max_arr, min_arr, avg_arr], axis=1)

def max_min_avg_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3  # only valid for 2D tensors
    shape[-2] = 3
    return tuple(shape)

###############################
### Create the main network ###
###############################
def create_main_sub_network(embedding_matrix):
    main_sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    embedded_sequences = embedding_layer(main_sequence_input)
    main_network = Conv1D(200, 3, border_mode="same", activation="tanh")(embedded_sequences)
    main_network = Lambda(max_min_avg_pooling_main, output_shape=max_min_avg_output_shape)(main_network)
    main_network = Flatten()(main_network)
    return (main_network, main_sequence_input)

#####################################
### Create the contextual network ###
#####################################
def create_contextual_sub_network(embedding_matrix):
    context_sequence_input = Input(shape=(MAX_CONTEXTUAL_WORDS,), dtype='int32')
    context_embedding_layer = Embedding(embedding_matrix.shape[0],
                                        embedding_matrix.shape[1],
                                        weights=[embedding_matrix],
                                        input_length=MAX_CONTEXTUAL_WORDS,
                                        trainable=False)
    aux_embedded_sequences = context_embedding_layer(context_sequence_input)
    context_network = Lambda(max_min_avg_pooling_context, output_shape=max_min_avg_output_shape, name="custom_pooling")(aux_embedded_sequences)
    context_network = Flatten()(context_network)
    return (context_network, context_sequence_input)

########################################
### Training and Evaluation Function ###
########################################
def train_and_display_metrics(model, x_train_arr, y_train, x_test_arr, y_test):
    model.fit(x_train_arr, y_train, validation_data=(x_test_arr, y_test),
              nb_epoch=40, batch_size=128, verbose=1)

    predicted_probabilities = model.predict(x_test_arr, batch_size=128, verbose=1)
    predicted_arr = predicted_probabilities.argmax(axis=1)

    actual_arr = y_test.argmax(axis=1)

    from sklearn import metrics
    print('Accuracy: {}\n'.format(metrics.accuracy_score(actual_arr, predicted_arr)))
    print(metrics.classification_report(actual_arr, predicted_arr))
    print(np.array_str(metrics.confusion_matrix(actual_arr, predicted_arr))) # ordering is alphabetical order of label names


########################################################################################################################

############################################################
### Function for testing the neural network with context ###
############################################################
def test_with_context():

    #################
    ### Load Data ###
    #################
    (x_train, y_train, x_conv_train, x_test, y_test, x_conv_test, embedding_matrix) = load_vanzo_dataset()
    print('Train: {} - Test: {} .'.format(len(x_train), len(x_test)))

    #################################
    ### Create the Neural Network ###
    #################################
    (main_network, main_sequence_input) = create_main_sub_network(embedding_matrix)
    (context_network, context_sequence_input) = create_contextual_sub_network(embedding_matrix)
    main_network = merge([main_network, context_network], mode='concat')
    main_network = Dense(64, activation='tanh')(main_network)
    predictions = Dense(3, activation='softmax')(main_network)

    model = Model(input=[main_sequence_input, context_sequence_input], output=[predictions])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=['acc'])

    ############################################
    ### Display some info about the networks ###
    ############################################
    print(model.summary())
    from keras.utils.visualize_util import plot
    plot(model, to_file='with_context_model.png', show_shapes=True)

    ######################
    ### Evaluate Model ###
    ######################
    train_and_display_metrics(model, [x_train, x_conv_train], y_train, [x_test, x_conv_test], y_test)

    ##################
    ### Save Model ###
    ##################
    with open("keras_model_with_context.json", "w") as json_file:
        json_file.write(model.to_json())
        json_file.close()
    model.save_weights("keras_model_with_context_weights.h5")


###############################################################
### Function for testing the neural network without context ###
###############################################################
def test_without_context():

    #################
    ### Load Data ###
    #################
    (x_train, y_train, x_conv_train, x_test, y_test, x_conv_test, embedding_matrix) = load_vanzo_dataset()
    print('Train: {} - Test: {} .'.format(len(x_train), len(x_test)))

    #################################
    ### Create the Neural Network ###
    #################################
    (main_network, main_sequence_input) = create_main_sub_network(embedding_matrix)
    main_network = Dense(64, activation='tanh')(main_network)
    predictions = Dense(3, activation='softmax')(main_network)

    model = Model(input=[main_sequence_input], output=[predictions])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=['acc'])

    ############################################
    ### Display some info about the networks ###
    ############################################
    print(model.summary())
    from keras.utils.visualize_util import plot
    plot(model, to_file='with_context_model.png', show_shapes=True)

    ######################
    ### Evaluate Model ###
    ######################
    train_and_display_metrics(model, [x_train], y_train, [x_test], y_test)

    ##################
    ### Save Model ###
    ##################
    with open("keras_model_no_context.json", "w") as json_file:
        json_file.write(model.to_json())
        json_file.close()
    model.save_weights("keras_model_no_context_weights.h5")


###################################
##### Actual Training Program #####
###################################

test_with_context()
test_without_context()