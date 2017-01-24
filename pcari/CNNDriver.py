import numpy as np
from keras.engine import Input, Model
from keras.layers import Embedding, Conv1D, Lambda, Flatten, Dense

YOLANDA_NOV2013_FEB2014_NPZ_PATH = "C:/Users/user/PycharmProjects/ms-thesis/pcari/yolanda_nov2013_feb2014.npz"

def load_dataset(dataset_path):
    data = np.load(YOLANDA_NOV2013_FEB2014_NPZ_PATH)

    #TODO update this when there is already a proper partitioning
    x_train = data["x"]
    y_train = data["y"]

    x_test = data["x"]
    y_test = data["y"]

    embedding_matrix = data["embedding_matrix"]

    return (x_train, y_train, x_test, y_test, embedding_matrix)

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
                                trainable=False)
    embedded_sequences = embedding_layer(main_sequence_input)
    main_network = Conv1D(200, 3, border_mode="same", activation="tanh")(embedded_sequences)
    main_network = Lambda(max_min_avg_pooling_main, output_shape=max_min_avg_output_shape)(main_network)
    main_network = Flatten()(main_network)
    main_network = Dense(64, activation='tanh')(main_network)
    predictions = Dense(6, activation='softmax')(main_network)

    model = Model(input=[main_sequence_input], output=[predictions])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=['acc'])
    return model



########################################
### Training and Evaluation Function ###
########################################
def train_and_display_metrics(model, x_train_arr, y_train, x_test_arr, y_test):
    model.fit(x_train_arr, y_train, validation_data=(x_test_arr, y_test),
              nb_epoch=10, batch_size=128, verbose=1)

    predicted_probabilities = model.predict(x_test_arr, batch_size=128, verbose=1)
    predicted_arr = predicted_probabilities.argmax(axis=1)

    actual_arr = y_test.argmax(axis=1)

    from sklearn import metrics
    print('Accuracy: {}\n'.format(metrics.accuracy_score(actual_arr, predicted_arr)))
    print(metrics.classification_report(actual_arr, predicted_arr))
    print(np.array_str(metrics.confusion_matrix(actual_arr, predicted_arr))) # ordering is alphabetical order of label names



def main_driver():
    (x_train, y_train, x_test, y_test, embedding_matrix) = load_dataset(YOLANDA_NOV2013_FEB2014_NPZ_PATH)
    model = create_model(embedding_matrix)
    train_and_display_metrics(model, x_train, y_train, x_test, y_test)

main_driver()