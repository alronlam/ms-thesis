import nltk
import math
import random
from foldersio import FolderIO
from csv_parser import CSVParser
from csv_parser.CSVParser import CSVParser
from nltk.classify.naivebayes import NaiveBayesClassifier
from sentiment_analysis.feature_extraction.UnigramExtractor import UnigramExtractor
from sentiment_analysis.classifier.ClassifierIO import *
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.classify.scikitlearn import SklearnClassifier
from sentiment_analysis.preprocessing.PreProcessing import *
from sentiment_analysis.feature_extraction import FeatureExtractorBase

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

    row_generator = CSVParser.parse_file_into_csv_row_generator('sa_training_data/globe_dataset.csv')
    tweet_texts = [row[2] for row in row_generator][:LIMIT]

    row_generator = CSVParser.parse_file_into_csv_row_generator('sa_training_data/globe_dataset.csv')
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


def test_senti_election_data():
    print("INSIDE HERE")
    csv_files = FolderIO.get_files('D:/DLSU/Masters/MS Thesis/data-2016/senti_election_data/csv_files', False, '.csv')
    print("FINISHED READING CSV FILES")
    print(csv_files.__len__())
    csv_rows = CSVParser.parse_files_into_csv_row_generator(csv_files)
    print("FINISHED READIN CSV ROWS")
    for csv_row in csv_rows:
        print(csv_row)
test_senti_election_data()

