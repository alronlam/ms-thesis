from datetime import datetime
from collections import Counter

import numpy as np
import sklearn
from keras.engine import Input, Model
from keras.layers import Embedding, Conv1D, Lambda, Flatten, Dense, MaxPooling1D
from sklearn.cross_validation import StratifiedKFold

YOLANDA_NOV2013_FEB2014_NPZ_PATH = "C:/Users/user/PycharmProjects/ms-thesis/pcari/data/relevant_irrelevant.npz"
NUM_CATEGORIES = 2

def load_dataset(dataset_path):
    data = np.load(dataset_path)
    return (data["x"], data["y"], data["embedding_matrix"])

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


def max_min_avg_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3  # only valid for 2D tensors
    shape[-2] = 3
    return tuple(shape)

###############################
### Create the main network ###
###############################
def create_model(embedding_matrix, MAX_SEQUENCE_LENGTH=32):
    main_sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    embedded_sequences = embedding_layer(main_sequence_input)
    main_network = Conv1D(200, 2, border_mode="same", activation="tanh")(embedded_sequences)
    main_network = MaxPooling1D(MAX_SEQUENCE_LENGTH)(main_network)
    # main_network = Lambda(max_min_avg_pooling_main, output_shape=max_min_avg_output_shape)(main_network)
    main_network = Flatten()(main_network)
    main_network = Dense(64, activation='tanh')(main_network)
    predictions = Dense(NUM_CATEGORIES, activation='softmax')(main_network)

    model = Model(input=[main_sequence_input], output=[predictions])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=['acc'])
    return model



########################################
### Training and Evaluation Function ###
########################################
def train_and_display_metrics(model, x_train_arr, y_train, x_test_arr, y_test, results_file):
    model.fit(x_train_arr, y_train, validation_data=(x_test_arr, y_test),
              nb_epoch=10, batch_size=128, verbose=1)

    predicted_probabilities = model.predict(x_test_arr, batch_size=128, verbose=1)
    predicted_arr = predicted_probabilities.argmax(axis=1)

    actual_arr = y_test.argmax(axis=1)

    from sklearn import metrics
    print('Accuracy: {}\n'.format(metrics.accuracy_score(actual_arr, predicted_arr)), file=results_file)
    print('Kappa: {}\n'.format(sklearn.metrics.cohen_kappa_score(actual_arr, predicted_arr)), file=results_file)
    print(metrics.classification_report(actual_arr, predicted_arr, digits=4), file=results_file)
    print(np.array_str(metrics.confusion_matrix(actual_arr, predicted_arr)), file=results_file) # ordering is alphabetical order of label names
    print("", file=results_file, flush=True)

    return model

def display_dataset_statistics(train_labels, test_labels, results_file):
    print("Train Data Composition({}):".format(len(train_labels)), file=results_file)
    counter = Counter(train_labels)
    print(counter.keys(), file=results_file)
    print(counter.values(), file=results_file)

    print("", file=results_file)

    print("Test Data Composition({}):".format(len(test_labels)), file=results_file)
    counter = Counter(test_labels)
    print(counter.keys(), file=results_file)
    print(counter.values(), file=results_file, flush=True)
    print("", file=results_file)

def main_driver(n_folds, data_dir, category, output_folder=""):
    results_file = open(output_folder+"results-{}-{}_folds-{}.txt".format(category, n_folds, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), "w")

    (data, labels, embedding_matrix) = load_dataset(data_dir)
    model = create_model(embedding_matrix)
    # train_and_display_metrics(model, x_train, y_train, x_test, y_test)

    numeric_labels = labels.argmax(axis=1)
    skf = StratifiedKFold(numeric_labels, n_folds=n_folds, shuffle=True)

    for i, (train, test) in enumerate(skf):
            print ("***********************\nRunning Fold {} / {}\n".format(i+1, n_folds), file=results_file)
            display_dataset_statistics(numeric_labels[train], numeric_labels[test], results_file)
            model = None # Clearing the NN.
            model = create_model(embedding_matrix)
            model = train_and_display_metrics(model, data[train], labels[train], data[test], labels[test], results_file)

            with open("{}_model_fold_{}.json".format(category, i), "w") as json_file:
                json_file.write(model.to_json())
                json_file.close()
            model.save_weights("{}_weights_fold_{}.h5".format(category, i))


# input_root_folder = "data/binary/npz/"
# output_root_folder="results/{}/".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
#
# categories = ["victim_identification_assistance", "raising_funds", "accounting_damage", "expressing_appreciation", "celebrification"]
# for category in categories:
#     main_driver(5, input_root_folder+category+".npz", category, "")


main_driver(3, YOLANDA_NOV2013_FEB2014_NPZ_PATH, "relevant_irrelevant")