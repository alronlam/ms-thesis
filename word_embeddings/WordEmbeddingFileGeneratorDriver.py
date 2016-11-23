import pickle

import math
import random

import numpy

from sentiment_analysis.evaluation import TSVParser
from twitter_data.database import DBManager
from twitter_data.parsing.folders import FolderIO
from word_embeddings import GoogleWordEmbedder


def shuffle_dataset_balanced(X, Y, test_percentage):
    target_num_test_instances = math.floor(len(X) * test_percentage)

    # stores the count for negative, neutral, and positive respectively
    # TODO: automatically extract this from unique values in the classes list
    test_classes_count = [0,0,0]
    target_limit_per_class = math.ciel(target_num_test_instances/3)

    test_X = []
    test_Y = []

    while sum(test_classes_count) < target_num_test_instances:
        random_index = random.randint(0, len(X))
        random_tweet = X[random_index]
        random_tweet_class = Y[random_index]

        if test_classes_count[random_tweet_class] < target_limit_per_class:
            test_X.append(random_tweet)
            test_Y.append(random_tweet_class)

            X.remove(random_index)
            Y.remove(random_index)

    train_X = X
    train_Y = Y

    return (train_X, train_Y, test_X, test_Y)



def save_dataset_to_embedding_file_multiclass(tweet_objects, classes, x_file_name):

    X = []
    Y = []
    for index, tweet_object in enumerate(tweet_objects):
        if tweet_object:
            print("{}".format(index))

            if classes[index] == 'negative':
                y = 0
            elif classes[index] == 'neutral':
                y = 1
            elif classes[index] == 'positive':
                y = 2

            embedded_word = GoogleWordEmbedder.google_embedding(tweet_object.text)

            X.append(embedded_word)
            Y.append(y)

    numpy.savez(x_file_name, X=X, Y=Y)


def save_driver_code():
    tsv_files = FolderIO.get_files('D:/DLSU/Masters/MS Thesis/data-2016/Context-Based_Tweets/test', True, '.tsv')
    conversations = TSVParser.parse_files_into_conversation_generator(tsv_files)

    print("CONSTRUCTING LISTS")
    tweet_objects = []
    classes = []
    for index, conversation in enumerate(conversations):
        target_tweet = conversation[-1]
        print("{} - {}".format(index, target_tweet["tweet_id"]))
        tweet_object = DBManager.get_or_add_tweet(target_tweet["tweet_id"])
        if tweet_object:
            tweet_objects.append(tweet_object)
            classes.append(target_tweet["class"])

    print("ENTERING FILE WRITING FUNCTION")
    save_dataset_to_embedding_file(tweet_objects, classes, "vanzo_dataset.npz")


save_driver_code()