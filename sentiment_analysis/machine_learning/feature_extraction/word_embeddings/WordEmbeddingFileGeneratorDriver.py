import pickle

import numpy
from keras.utils.np_utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer

from community_detection import Utils
from sentiment_analysis.evaluation import TSVParser
from sentiment_analysis.machine_learning.feature_extraction.word_embeddings import GoogleWordEmbedder
from sentiment_analysis.preprocessing import PreProcessing
from sentiment_analysis.preprocessing.PreProcessing import SplitWordByWhitespace, ReplaceURL, RemovePunctuationFromWords, \
    ReplaceUsernameMention, RemoveRT, RemoveLetterRepetitions, ConcatWordArray, RemoveExactTerms
from sentiment_analysis.preprocessing.PreProcessing import WordToLowercase
from twitter_data.database import DBManager
from twitter_data.parsing.folders import FolderIO

#TODO: This is not verified to be working correctly
# def shuffle_dataset_balanced(X, Y, test_percentage):
#     target_num_test_instances = math.floor(len(X) * test_percentage)
#
#     # stores the count for negative, neutral, and positive respectively
#     # TODO: automatically extract this from unique values in the classes list
#     test_classes_count = [0,0,0]
#     target_limit_per_class = math.ceil(target_num_test_instances/3)
#
#     test_X = []
#     test_Y = []
#
#     while sum(test_classes_count) < target_num_test_instances and len(X) > 0:
#         random_index = random.randint(0, len(X)-1)
#         random_tweet = X[random_index]
#         random_tweet_class = Y[random_index]
#
#         if test_classes_count[random_tweet_class] < target_limit_per_class:
#             test_X.append(random_tweet)
#             test_Y.append(random_tweet_class)
#             test_classes_count[random_tweet_class] += 1
#
#             X.__delitem__(random_index)
#             Y.__delitem__(random_index)
#
#             print(test_classes_count)
#     train_X = X
#     train_Y = Y
#
#     print("Train: {}, Test: {} ({})".format(len(train_X), len(test_X), test_classes_count))
#
#     return (train_X, train_Y, test_X, test_Y)


##### SOME CONSTANTS #####
VANZO_TRAIN_DIR = 'D:/DLSU/Masters/MS Thesis/data-2016/Context-Based_Tweets/conv_train'
VANZO_TEST_DIR = 'D:/DLSU/Masters/MS Thesis/data-2016/Context-Based_Tweets/conv_test'


##### UTILITY FUNCTIONS #####
def convert_sentiment_class_to_number(sentiment_class):
    sentiment_class = sentiment_class.lower()
    if sentiment_class == "negative":
        return 0
    if sentiment_class == "neutral":
        return 1
    if sentiment_class == "positive":
        return 2

def load_tsv_dataset(path):
    texts = []
    labels = []
    contextual_tweets = []
    tsv_files = FolderIO.get_files(path, True, '.tsv')
    conversations = TSVParser.parse_files_into_conversation_generator(tsv_files)
    for index, conversation in enumerate(conversations):
        # actual tweet
        target_tweet = conversation[-1]
        tweet_object = DBManager.get_or_add_tweet(target_tweet["tweet_id"])
        if tweet_object:
            tweet_class = target_tweet["class"]
            tweet_text = tweet_object.text

            texts.append(tweet_text)
            labels.append(convert_sentiment_class_to_number(tweet_class))

            # generate the list of conversational tweets for the target tweet
            curr_contextual_tweets = []
            for contextual_tweet in conversation[:-1]:
                tweet_object = DBManager.get_or_add_tweet(contextual_tweet["tweet_id"])
                if tweet_object:
                    curr_contextual_tweets.append(tweet_object.text)

            # append the current list of conversational tweets to the overall list
            contextual_tweets.append(curr_contextual_tweets)

        print(index)

    return (texts, labels, contextual_tweets)



##### Google Word Embedding Functions - Output can be used directly as word vectors #####
def generate_npz_avg_embedding(source_dir, npz_file_name ):
    (texts, labels) = load_tsv_dataset(source_dir)

    print("CONSTRUCTING LISTS")
    X = []
    Y = []

    for index in range(len(texts)):
        X.append(GoogleWordEmbedder.google_embedding_avg(texts[index]))
        Y.append(convert_sentiment_class_to_number(labels[index]))

    print("ENTERING FILE WRITING FUNCTION")
    numpy.savez(npz_file_name, X=X, Y=Y)


# generate_npz_avg_embedding(VANZO_TRAIN_DIR, 'vanzo_train_avg.npz')
# generate_npz_avg_embedding(VANZO_TEST_DIR, 'vanzo_test_avg.npz')


def generate_npz_concat_embedding(source_dir, npz_file_name, max_word_count):

    (texts, labels) = load_tsv_dataset(source_dir)

    print("CONSTRUCTING LISTS")
    X = []
    Y = []

    for index in range(len(texts)):
        X.append(GoogleWordEmbedder.google_embedding_concat(texts[index], max_word_count))
        Y.append(convert_sentiment_class_to_number(labels[index]))

    print("ENTERING FILE WRITING FUNCTION")
    numpy.savez(npz_file_name, X=X, Y=Y)


# generate_npz_concat_embedding(VANZO_TRAIN_DIR, 'vanzo_train_concat.npz', 15)
# generate_npz_concat_embedding(VANZO_TEST_DIR, 'vanzo_test_concat.npz', 15)

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


def convert_contextual_tweets_by_idf(contextual_tweets, TOP_N_KEYWORDS):

    top_keywords = []

    for curr_contextual_tweets in contextual_tweets:
        try:
            tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
            tfidf_vectorizer.fit_transform(curr_contextual_tweets)

            indices = numpy.argsort(tfidf_vectorizer.idf_)[::-1]
            features = tfidf_vectorizer.get_feature_names()
            top_features = [features[i] for i in indices[:TOP_N_KEYWORDS]]
            print(top_features)
        except Exception as e:
            print(e)
            top_features = []

        top_feature_string = " ".join(top_features)
        top_keywords.append(top_feature_string)

    return top_keywords


def generate_npz_word_index_sequence(train_dir, test_dir, npz_file_name, MAX_NB_WORDS = 20000, MAX_SEQUENCE_LENGTH = 32, TOP_N_KEYWORDS=32):

    (x_train, y_train, x_conv_train) = load_tsv_dataset(train_dir)
    (x_test, y_test, x_conv_test) = load_tsv_dataset(test_dir)

    # Pre-process text
    target_tweet_preprocessors = [SplitWordByWhitespace(),
                 WordToLowercase(),
                 ReplaceURL(),
                 RemovePunctuationFromWords(),
                 ReplaceUsernameMention(),
                 RemoveRT(),
                 RemoveLetterRepetitions(),
                 ConcatWordArray()]

    conv_context_preprocessors = [SplitWordByWhitespace(),
                 WordToLowercase(),
                 ReplaceURL(),
                 RemovePunctuationFromWords(),
                 ReplaceUsernameMention(),
                 RemoveRT(),
                 RemoveLetterRepetitions(),
                 RemoveExactTerms(Utils.load_function_words("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/preprocessing/eng-function-words.txt")),
                 ConcatWordArray()]

    x_train = PreProcessing.preprocess_strings(x_train, target_tweet_preprocessors)
    x_test = PreProcessing.preprocess_strings(x_test, target_tweet_preprocessors)

    for index, conv_context in enumerate(x_conv_train):
        x_conv_train[index] = PreProcessing.preprocess_strings(conv_context, conv_context_preprocessors)
    for index, conv_context in enumerate(x_conv_test):
        x_conv_test[index] = PreProcessing.preprocess_strings(conv_context, conv_context_preprocessors)


    x_conv_train = convert_contextual_tweets_by_idf(x_conv_train, TOP_N_KEYWORDS)
    x_conv_test = convert_contextual_tweets_by_idf(x_conv_test, TOP_N_KEYWORDS)

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(x_train + x_conv_train)
    train_sequences = tokenizer.texts_to_sequences(x_train)
    test_sequences = tokenizer.texts_to_sequences(x_test)
    train_conv_sequences = tokenizer.texts_to_sequences(x_conv_train)
    test_conv_sequences = tokenizer.texts_to_sequences(x_conv_test)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    x_train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    x_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    x_conv_train = pad_sequences(train_conv_sequences, maxlen=TOP_N_KEYWORDS)
    x_conv_test = pad_sequences(test_conv_sequences, maxlen=TOP_N_KEYWORDS)

    y_train = to_categorical(numpy.asarray(y_train))
    y_test = to_categorical(numpy.asarray(y_test))

    print('Shape of train data tensor:', x_train.shape)
    print('Shape of train label tensor:', y_train.shape)
    print('Shape of train conv data tensor:', x_conv_train.shape)

    print('Shape of test data tensor:', x_test.shape)
    print('Shape of test label tensor:', y_test.shape)
    print('Shape of test conv data tensor:', x_conv_test.shape)

    print('Train: {} - Test: {} .'.format(len(x_train), len(x_test)))

    pickle.dump(tokenizer, open("tokenizer-{}.pickle".format(npz_file_name), "wb"))
    embedding_matrix = generate_glove_embedding_matrix(tokenizer.word_index)
    numpy.savez(npz_file_name,
                x_train=x_train, y_train=y_train, x_conv_train=x_conv_train,
                x_test=x_test, y_test=y_test, x_conv_test=x_conv_test,
                embedding_matrix=embedding_matrix)


generate_npz_word_index_sequence(VANZO_TRAIN_DIR, VANZO_TEST_DIR, 'vanzo_word_sequence_concat_glove_200d_preprocessed.npz')

