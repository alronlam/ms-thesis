import math
import random
import pickle
from datetime import datetime
from os import path

import nltk
import numpy
import sklearn

from sentiment_analysis.evaluation import TSVParser
from sentiment_analysis.machine_learning.feature_extraction import FeatureExtractorBase
from sklearn import metrics

from sentiment_analysis.preprocessing import PreProcessing
from twitter_data.database import DBManager
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
    tweet_texts = preprocess_strings(tweet_texts, TWEET_PREPROCESSORS)

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


# test_nltk()

def read_data(source_dir, file_extension):
    dataset_files = FolderIO.get_files(source_dir, False, file_extension)
    conversation_generator = TSVParser.parse_files_into_conversation_generator(dataset_files)
    X = []
    Y = []
    for index, conversation in enumerate(conversation_generator):
        target_tweet = conversation[-1]
        tweet_id = target_tweet["tweet_id"]
        tweet_object = DBManager.get_or_add_tweet(tweet_id)
        if tweet_object and tweet_object.text:
            X.append(tweet_object.text)
            Y.append(target_tweet["class"])

    return (X,Y)

def train_sa_with_unigrams():
    print("WENT IN")
    # read data
    (X_train,Y_train) = read_data('D:/DLSU/Masters/MS Thesis/data-2016/Context-Based_Tweets/conv_train', '.tsv')
    (X_test, Y_test ) = read_data('D:/DLSU/Masters/MS Thesis/data-2016/Context-Based_Tweets/conv_test', '.tsv')

    # pre-process tweets
    TWEET_PREPROCESSORS = [SplitWordByWhitespace(), WordLengthFilter(3), RemovePunctuationFromWords(), WordToLowercase()]
    X_train = PreProcessing.preprocess_strings(X_train, TWEET_PREPROCESSORS)
    X_test = PreProcessing.preprocess_strings(X_test, TWEET_PREPROCESSORS)
    print("FINISHED PREPROCESSING")

    # construct labeled tweets to be run with the classifiers
    train_tweets = list(zip(X_train, Y_train))
    test_tweets = list(zip(X_test, Y_test))

    print("# TRAIN: {}".format(train_tweets.__len__()))
    print("# TEST: {}".format(test_tweets.__len__()))

    # feature extraction
    FEATURE_EXTRACTOR = UnigramExtractor(train_tweets, 1000)
    FeatureExtractorBase.save_feature_extractor("sa_svm_unigram_feature_extractor_vanzo_conv_train.pickle", FEATURE_EXTRACTOR)
    training_set = nltk.classify.apply_features(FEATURE_EXTRACTOR.extract_features, train_tweets)
    print("FINISHED EXTRACTING FEATURES FROM TWEETS")

    # training
    # trainer = nltk.NaiveBayesClassifier
    trainer = nltk.classify.SklearnClassifier(sklearn.svm.LinearSVC())
    # trainer = SklearnClassifier(BernoulliNB())
    classifier = trainer.train(training_set)
    pickle.dump(classifier, open( "sa_svm_classifier_vanzo_conv_train.pickle", "wb"))
    # print(classifier.show_most_informative_features(15))

    #classification
    test_set = nltk.classify.apply_features(FEATURE_EXTRACTOR.extract_features, test_tweets)
    print(nltk.classify.accuracy(classifier, test_set))

train_sa_with_unigrams()