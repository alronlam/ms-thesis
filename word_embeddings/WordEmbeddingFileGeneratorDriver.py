import pickle

import numpy

from sentiment_analysis.evaluation import TSVParser
from twitter_data.database import DBManager
from twitter_data.parsing.folders import FolderIO
from word_embeddings import GoogleWordEmbedder


def save_dataset_to_embedding_file(tweet_objects, classes, x_file_name):

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