import numpy as np
from itertools import groupby

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


(x_train, y_train, x_conv_train, x_test, y_test, x_conv_test, embedding_matrix) = load_vanzo_dataset()
y_train = y_train.argmax(axis=1)
y_test = y_test.argmax(axis=1)

from collections import Counter
print(Counter(y_train))
print(Counter(y_test))


x_conv_train = [y_train[index] if np.count_nonzero(x_conv) > 0 else None for index, x_conv in enumerate(x_conv_train)]
x_conv_test = [y_test[index] if np.count_nonzero(x_conv) > 0 else None for index, x_conv in enumerate(x_conv_test)]

print(Counter(x_conv_train))
print(Counter(x_conv_test))

