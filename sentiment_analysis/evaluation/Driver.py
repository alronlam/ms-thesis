from sentiment_analysis.evaluation import TSVParser
from sentiment_analysis import SentimentClassifier
from twitter_data.parsing.folders import FolderIO
from twitter_data.database import DBManager

import pickle
from datetime import datetime

import numpy
from sklearn import metrics

def write_metrics_file(actual_arr, predicted_arr, metrics_file_name):
    metrics_file = open(metrics_file_name, 'w')
    metrics_file.write('Total: {}\n'.format(actual_arr.__len__()))
    metrics_file.write('Accuracy: {}\n'.format(metrics.accuracy_score(actual_arr, predicted_arr)))
    try:
        metrics_file.write(metrics.classification_report(actual_arr, predicted_arr))
    except Exception as e:
        print(e)
        pass
    metrics_file.write('\n')
    metrics_file.write(numpy.array_str(metrics.confusion_matrix(actual_arr, predicted_arr))) # ordering is alphabetical order of label names
    metrics_file.write('\n')
    metrics_file.close()

def test_vanzo_eng_dataset(classifier):

    tsv_files = FolderIO.get_files('D:/DLSU/Masters/MS Thesis/data-2016/Context-Based_Tweets/test', True, '.tsv')
    conversations = TSVParser.parse_files_into_conversation_generator(tsv_files)

    actual_arr = []
    predicted_arr = []

    metrics_file_name = 'metrics-vanzo-eng-{}-afinn.txt'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    for index, conversation in enumerate(conversations):
        target_tweet = conversation[-1]

        print("{} - {}".format(index, target_tweet["tweet_id"]))
        tweet_object = DBManager.get_or_add_tweet(target_tweet["tweet_id"])
        if tweet_object:
            predicted_class = classifier.classify_sentiment(tweet_object.text)
            actual_class = target_tweet["class"]

            predicted_arr.append(predicted_class)
            actual_arr.append(actual_class)

            # print("{} vs {}".format( actual_class,predicted_class))

        if index % 100 == 0 and index > 0:
            pickle.dump((actual_arr, predicted_arr), open( "{}.pickle".format(metrics_file_name), "wb" ) )
            write_metrics_file(actual_arr, predicted_arr, metrics_file_name)


    pickle.dump((actual_arr, predicted_arr), open( "{}.pickle".format(metrics_file_name), "wb" ) )
    write_metrics_file(actual_arr, predicted_arr, metrics_file_name)


lexicon_classifier = SentimentClassifier.LexiconClassifier()
anew_lexicon_classifier = SentimentClassifier.ANEWLexiconClassifier()
globe_ml_classifier = SentimentClassifier.MLClassifier("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/unigram_feature_extractor.pickle", "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/nb_classifier.pickle.pickle")
afinn_classifier = SentimentClassifier.AFINNLexiconClassifier()

test_vanzo_eng_dataset(afinn_classifier)
# test_vanzo_eng_dataset(anew_lexicon_classifier)
# test_vanzo_eng_dataset(lexicon_classifier)
# test_vanzo_eng_dataset(globe_ml_classifier)
