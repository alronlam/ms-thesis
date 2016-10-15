import math
import random
import pickle
from datetime import datetime
from os import path

import nltk
import numpy
from sentiment_analysis.machine_learning.feature_extraction import FeatureExtractorBase
from sklearn import metrics

from twitter_data.parsing.csv_parser import CSVParser
from twitter_data.parsing.folders import FolderIO
from sentiment_analysis import SentimentClassifier
from sentiment_analysis.machine_learning.feature_extraction.UnigramExtractor import UnigramExtractor
from sentiment_analysis.preprocessing.PreProcessing import *

def save_classifier_to_pickle(pickle_file_name, classifier):
    pickle.dump(classifier, open( "{}.pickle".format(pickle_file_name), "wb"))

def load_classifier_from_pickle(pickle_file_name):
    return pickle.load(pickle_file_name)


def train_or_load(pickle_file_name, trainer, training_set, force_train=False):
    classifier = None
    if not force_train:
        classifier = load_classifier_from_pickle(pickle_file_name)
    if not classifier:
        classifier = trainer.train(training_set)
        save_classifier_to_pickle(pickle_file_name, classifier)
    return classifier

def test_nltk():
    # read data
    LIMIT = 213

    dataset_files = FolderIO.get_files('C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/sa_training_data/', False, '.csv')
    row_generator = CSVParser.parse_files_into_csv_row_generator(dataset_files, False)
    tweet_texts = [row[2] for row in row_generator][:LIMIT]

    dataset_files = FolderIO.get_files('C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/sa_training_data/', False, '.csv')
    row_generator = CSVParser.parse_files_into_csv_row_generator(dataset_files, False)
    tweet_labels = [row[1] for row in row_generator][:LIMIT]

    # pre-process tweets
    TWEET_PREPROCESSORS = [SplitWordByWhitespace(), WordLengthFilter(3), WordToLowercase()]
    tweet_texts = preprocess_tweets(tweet_texts, TWEET_PREPROCESSORS)

    # construct labeled tweets to be run with the classifiers
    labeled_tweets = list(zip(tweet_texts, tweet_labels))


    # partition training/testing sets
    random.shuffle(labeled_tweets) # shuffling here to randomize train and test tweets
    num_train = math.floor(LIMIT * 0.6)
    train_tweets = labeled_tweets[:num_train]
    test_tweets = labeled_tweets[num_train:]

    print("# TRAIN: {}".format(train_tweets.__len__()))
    print("# TEST: {}".format(test_tweets.__len__()))

    # feature extraction
    FEATURE_EXTRACTOR = UnigramExtractor(train_tweets)
    FeatureExtractorBase.save_feature_extractor("unigram_feature_extractor.pickle", FEATURE_EXTRACTOR)
    training_set = nltk.classify.apply_features(FEATURE_EXTRACTOR.extract_features, train_tweets)

    # training
    TRAINER = nltk.NaiveBayesClassifier
    # TRAINER = SklearnClassifier(BernoulliNB())
    classifier = train_or_load("nb_classifier.pickle", TRAINER, training_set, True)
    print(classifier.show_most_informative_features(15))

    #classification
    test_set = nltk.classify.apply_features(FEATURE_EXTRACTOR.extract_features, test_tweets)
    print(nltk.classify.accuracy(classifier, test_set))


test_nltk()