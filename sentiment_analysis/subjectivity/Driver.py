import math
import random
import pickle
from datetime import datetime
from os import path

from sentiment_analysis.evaluation import TSVParser
from twitter_data.database import DBManager

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
    dataset_files = FolderIO.get_files('D:/DLSU/Masters/MS Thesis/data-2016/Context-Based_Tweets/test', False, '.tsv')
    conversation_generator = TSVParser.parse_files_into_conversation_generator(dataset_files)
    tweet_texts = []
    tweet_labels = []
    for index, conversation in enumerate(conversation_generator):
        target_tweet = conversation[-1]
        tweet_id = target_tweet["tweet_id"]
        tweet_object = DBManager.get_or_add_tweet(tweet_id)
        if tweet_object and tweet_object.text:
            tweet_texts.append(tweet_object.text)
            tweet_labels.append(target_tweet["class"])
            print("Constructing data lists. At index {}".format(index))

    # pre-process tweets
    TWEET_PREPROCESSORS = [SplitWordByWhitespace(), WordLengthFilter(3), WordToLowercase()]
    tweet_texts = preprocess_tweets(tweet_texts, TWEET_PREPROCESSORS)
    print("FINISHED PREPROCESSING")
    # construct labeled tweets to be run with the classifiers
    labeled_tweets = list(zip(tweet_texts, tweet_labels))

    # partition training/testing sets
    random.shuffle(labeled_tweets) # shuffling here to randomize train and test tweets
    num_train = math.floor(tweet_texts.__len__() * 0.9)
    train_tweets = labeled_tweets[:num_train]
    test_tweets = labeled_tweets[num_train:]

    print("# TRAIN: {}".format(train_tweets.__len__()))
    print("# TEST: {}".format(test_tweets.__len__()))

    # feature extraction
    FEATURE_EXTRACTOR = UnigramExtractor(train_tweets)
    FeatureExtractorBase.save_feature_extractor("subj_unigram_feature_extractor.pickle", FEATURE_EXTRACTOR)
    training_set = nltk.classify.apply_features(FEATURE_EXTRACTOR.extract_features, train_tweets)

    # training
    TRAINER = nltk.NaiveBayesClassifier
    # TRAINER = SklearnClassifier(BernoulliNB())
    classifier = train_or_load("subj_nb_classifier.pickle", TRAINER, training_set, True)
    print(classifier.show_most_informative_features(15))

    #classification
    test_set = nltk.classify.apply_features(FEATURE_EXTRACTOR.extract_features, test_tweets)
    print(nltk.classify.accuracy(classifier, test_set))


test_nltk()