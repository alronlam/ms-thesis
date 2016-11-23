import pickle

import math
import random

import numpy

from sentiment_analysis.evaluation import TSVParser
from twitter_data.database import DBManager
from twitter_data.parsing.folders import FolderIO
from word_embeddings import GoogleWordEmbedder


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


def convert_sentiment_class_to_number(sentiment_class):
    sentiment_class = sentiment_class.lower()
    if sentiment_class == "negative":
        return 0
    if sentiment_class == "neutral":
        return 1
    if sentiment_class == "positive":
        return 2

def generate_npz(source_dir, file_extension, npz_file_name ):
    tsv_files = FolderIO.get_files(source_dir, True, file_extension)
    conversations = TSVParser.parse_files_into_conversation_generator(tsv_files)

    print("CONSTRUCTING LISTS")
    X = []
    Y = []
    for index, conversation in enumerate(conversations):
        target_tweet = conversation[-1]
        print("{} - {}".format(index, target_tweet["tweet_id"]))
        tweet_object = DBManager.get_or_add_tweet(target_tweet["tweet_id"])
        if tweet_object:
            X.append(GoogleWordEmbedder.google_embedding_avg(tweet_object.text))
            Y.append(convert_sentiment_class_to_number(target_tweet["class"]))

    print("ENTERING FILE WRITING FUNCTION")
    # (train_X, train_Y, test_X, test_Y) = shuffle_dataset_balanced(X, Y, 0.3)
    # numpy.savez("vanzo_dataset_partitioned_balanced.npz", train_X=train_X, trainY=train_Y, test_X=test_X, test_Y=test_Y)

    numpy.savez(npz_file_name, X=X, Y=Y)


generate_npz('D:/DLSU/Masters/MS Thesis/data-2016/Context-Based_Tweets/conv_train', '.tsv', 'vanzo_train.npz')
generate_npz('D:/DLSU/Masters/MS Thesis/data-2016/Context-Based_Tweets/conv_test', '.tsv', 'vanzo_test.npz')
