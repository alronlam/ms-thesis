import os
import pickle

from keras.engine import Model, Input, merge
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

def load_news_dataset():

    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    f = open(fpath)
                    texts.append(f.read())
                    f.close()
                    labels.append(label_id)

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_test = data[-nb_validation_samples:]
    y_test = labels[-nb_validation_samples:]

    return (x_train, y_train, x_test, y_test, word_index)


def load_vanzo_dataset():
    data = np.load("C:/Users/user/PycharmProjects/ms-thesis/word_embeddings/vanzo_word_sequence_concat.npz")
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_conv_train = data["x_conv_train"]

    x_test = data["x_test"]
    y_test = data["y_test"]
    x_conv_test = data["x_conv_test"]

    embedding_matrix = data["embedding_matrix"]

    # word_index = pickle.load(open("C:/Users/user/PycharmProjects/ms-thesis/word_embeddings/vanzo_word_sequence_concat.npz-word_index.pickle", "rb"))

    return (x_train, y_train, x_conv_train, x_test, y_test, x_conv_test, embedding_matrix)


#########################################################################################################


###################################
##### Actual Training Program #####
###################################



#################
### Load Data ###
#################

# (x_train, y_train, x_test, y_test, word_index) = load_news_dataset()
(x_train, y_train, x_conv_train, x_test, y_test, x_conv_test, embedding_matrix) = load_vanzo_dataset()

print('Train: {} - Test: {} .'.format(len(x_train), len(x_test)))


##############################################
### Create the Neural Network Architecture ###
##############################################

from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, AveragePooling1D

embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
# x = Conv1D(128, 5, activation='relu')(embedded_sequences)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
# x = MaxPooling1D(35)(x)  # global max pooling
x = Conv1D(1, 3, border_mode="valid", activation="tanh")(embedded_sequences)
x = AveragePooling1D(3)(x)
x = Flatten()(x)

aux_input = Input(shape=(MAX_CONTEXTUAL_WORDS,), dtype='int32')
contextual_embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
aux_embedded_sequences = contextual_embedding_layer(aux_input)
aux_network = AveragePooling1D(3)(aux_embedded_sequences)
aux_network = Flatten()(aux_network)

x = merge([x, aux_network], mode='concat')
x = Dense(64, activation='relu')(x)

preds = Dense(3, activation='softmax')(x)

model = Model(input=[sequence_input, aux_input], output=[preds])
model.compile(loss='categorical_crossentropy',
              optimizer='adagrad',
              metrics=['acc'])

print(model.summary())

from keras.utils.visualize_util import plot
plot(model, to_file='model.png', show_shapes=True)


print("X shape: {}".format(x_train.shape))
print("Y shape: {}".format(y_train.shape))



######################
### Evaluate Model ###
######################

model.fit([x_train,  x_conv_train], y_train, validation_data=([x_test, x_conv_test], y_test),
          nb_epoch=10, batch_size=128, verbose=1)

predicted_probabilities = model.predict([x_test, x_conv_test], batch_size=128, verbose=1)
predicted_arr = predicted_probabilities.argmax(axis=1)

actual_arr = y_test.argmax(axis=1)


from sklearn import metrics
print('Accuracy: {}\n'.format(metrics.accuracy_score(actual_arr, predicted_arr)))
print(metrics.classification_report(actual_arr, predicted_arr))
print(np.array_str(metrics.confusion_matrix(actual_arr, predicted_arr))) # ordering is alphabetical order of label names