from sentiment_analysis.evaluation import TSVParser
from sentiment_analysis import SentimentClassifier
from twitter_data.parsing.folders import FolderIO
from twitter_data.database import DBManager

from datetime import datetime

import numpy
from sklearn import metrics

def write_metrics_file(actual_arr, predicted_arr, metrics_file_name):
    metrics_file = open(metrics_file_name, 'w')
    metrics_file.write('Total: {}\n'.format(actual_arr.__len__()))
    metrics_file.write('Accuracy: {}\n'.format(metrics.accuracy_score(actual_arr, predicted_arr)))
    metrics_file.write(metrics.classification_report(actual_arr, predicted_arr))
    metrics_file.write('\n')
    metrics_file.write(numpy.array_str(metrics.confusion_matrix(actual_arr, predicted_arr))) # ordering is alphabetical order of label names
    metrics_file.write('\n')
    metrics_file.close()

def test_vanzo_eng_dataset(classifier):

    tsv_files = FolderIO.get_files('D:/DLSU/Masters/MS Thesis/data-2016/Context-Based_Tweets/test', True, '.tsv')
    conversations = TSVParser.parse_files_into_conversation_generator(tsv_files)

    actual_arr = []
    predicted_arr = []

    metrics_file_name = 'metrics-vanzo-eng-{}.txt'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

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
            write_metrics_file(actual_arr, predicted_arr, metrics_file_name)


test_vanzo_eng_dataset(classifier=SentimentClassifier.ANEWLexiconClassifier())

# print(DBManager.get_or_add_tweet("100002637658849280"))