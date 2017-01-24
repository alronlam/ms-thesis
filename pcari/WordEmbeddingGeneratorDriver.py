import csv
import pickle

import numpy
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing

##### SOME CONSTANTS #####
YOLANDA_NOV2013_FEB2014_CSV_FILE = "C:/Users/user/PycharmProjects/ms-thesis/pcari/data/yolanda_nov2013_feb2014_dataset.csv"

##### UTILITY FUNCTIONS #####
def create_label_encoder(labels):
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(labels)
    return label_encoder

##### Word Index Sequence Functions - output can be used with an embedding layer  #####


#######################################################
### Create Embedding Matrix for the Embedding Layer ###
#######################################################

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def generate_glove_embedding_matrix(word_index):

    GLOVE_DIR = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/feature_extraction/word_embeddings/glove/glove.twitter.27B.200d.txt"
    EMBEDDING_DIM = 200

    embeddings_index = {}
    f = open(GLOVE_DIR, errors='ignore')
    for line in f:
        try:
            values = line.split()
            word = values[0]
            coefs = numpy.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except:
            continue
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = numpy.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def load_csv_dataset(file_path):
    with open(file_path, "r", encoding="utf8", newline="") as csv_file:
        row_reader = csv.reader(csv_file, delimiter=',')
        dataset = [row for row in row_reader]

    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    return (x,y)

def generate_npz_word_index_sequence(data_dir, npz_file_name, MAX_NB_WORDS=20000, MAX_SEQUENCE_LENGTH=32):
    (x, y) = load_csv_dataset(data_dir)

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(x)

    x = tokenizer.texts_to_sequences(x)
    x = pad_sequences(x, maxlen=MAX_SEQUENCE_LENGTH)

    label_encoder = create_label_encoder(y)
    y = label_encoder.transform(y)
    y = to_categorical(numpy.asarray(y))

    pickle.dump(tokenizer, open("tokenizer-{}.pickle".format(npz_file_name), "wb"))
    embedding_matrix = generate_glove_embedding_matrix(tokenizer.word_index)
    numpy.savez(npz_file_name,
                x=x, y=y,
                embedding_matrix=embedding_matrix)


generate_npz_word_index_sequence(YOLANDA_NOV2013_FEB2014_CSV_FILE, 'yolanda_nov2013_feb2014.npz')

