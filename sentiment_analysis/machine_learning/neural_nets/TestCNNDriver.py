# load models
import numpy

from sentiment_analysis import SentimentClassifier
from sentiment_analysis.evaluation import TSVParser
from twitter_data.database import DBManager
from twitter_data.parsing.folders import FolderIO

keras_tokenizer_pickle_path = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/feature_extraction/word_embeddings/tokenizer-vanzo_word_sequence_concat_glove_200d_preprocessed.npz.pickle"
keras_classifier_json_path = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/neural_nets/keras_model_with_context.json"
keras_classifier_weights_path = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/neural_nets/keras_model_with_context_weights.h5"
keras_classifier = SentimentClassifier.KerasClassifier(keras_tokenizer_pickle_path, keras_classifier_json_path, keras_classifier_weights_path, with_context=True)

def convert_sentiment_class_to_number(sentiment_class):
    sentiment_class = sentiment_class.lower()
    if sentiment_class == "negative":
        return 0
    if sentiment_class == "neutral":
        return 1
    if sentiment_class == "positive":
        return 2

# load datasets
def load_vanzo_dataset():
    data = numpy.load("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/feature_extraction/word_embeddings/vanzo_word_sequence_concat_glove_200d_preprocessed.npz")
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_conv_train = data["x_conv_train"]

    x_test = data["x_test"]
    y_test = data["y_test"]
    x_conv_test = data["x_conv_test"]

    embedding_matrix = data["embedding_matrix"]

    return (x_train, y_train, x_conv_train, x_test, y_test, x_conv_test, embedding_matrix)

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

VANZO_TRAIN_DIR = 'D:/DLSU/Masters/MS Thesis/data-2016/Context-Based_Tweets/conv_train'
(texts, labels, contextual_tweets) = load_tsv_dataset(VANZO_TRAIN_DIR)
print("{} - {} - {}".format(len(texts), len(labels), len(contextual_tweets)))

# call predict, passing both target words and conversational words

predicted_arr = []
for index in range(len(texts)):
    text = texts[index]
    contextual_tweet_thread = contextual_tweets[index]

    print(index)
    predicted_arr.append(convert_sentiment_class_to_number(keras_classifier.classify_sentiment(text, {"conv_context":contextual_tweet_thread})))


(x_train, y_train, x_conv_train, x_test, y_test, x_conv_test, embedding_matrix) = load_vanzo_dataset()

actual_arr = y_train.argmax(axis=1)

from sklearn import metrics
print('Accuracy: {}\n'.format(metrics.accuracy_score(actual_arr, predicted_arr)))
print(metrics.classification_report(actual_arr, predicted_arr))
print(numpy.array_str(metrics.confusion_matrix(actual_arr, predicted_arr))) # ordering is alphabetical order of label names