import nltk
import math
import random
from csv_parser.CSVParser import CSVParser
from nltk.classify.naivebayes import NaiveBayesClassifier
from sentiment_analysis.feature_extraction.UnigramExtractor import UnigramExtractor
from sentiment_analysis.classifier.ClassifierIO import *
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.classify.scikitlearn import SklearnClassifier
from sentiment_analysis.preprocessing.PreProcessing import *

def train_or_load(pickle_file_name, trainer, training_set, force_train=False):
    classifier = None
    if not force_train:
        classifier = load_classifier_from_pickle(pickle_file_name)
    if not classifier:
        classifier = trainer.train(training_set)
        save_classifier_to_pickle(pickle_file_name, classifier)
    return classifier

# read data
LIMIT = 213

row_generator = CSVParser.parse_file_into_csv_row_generator('sa_training_data/globe_dataset.csv')

labeled_tweets = [([word for word in row[2].split()], row[1]) for row in row_generator]
labeled_tweets = labeled_tweets[0:LIMIT]

# pre-process tweets
TWEET_PREPROCESSORS = [WordLengthFilter(3), WordToLowercase()]
labeled_tweets = preprocess_tweets(labeled_tweets, TWEET_PREPROCESSORS)

# partition training/testing sets
random.shuffle(labeled_tweets) # shuffling here to randomize train and test tweets
num_train = math.floor(LIMIT * 0.6)
train_tweets = labeled_tweets[:num_train]
test_tweets = labeled_tweets[num_train:]

print("# TRAIN: {}".format(train_tweets.__len__()))
print("# TEST: {}".format(test_tweets.__len__()))

# feature extraction
FEATURE_EXTRACTOR = UnigramExtractor(train_tweets)
training_set = nltk.classify.apply_features(FEATURE_EXTRACTOR.extract_features, train_tweets)

# training
TRAINER = nltk.NaiveBayesClassifier
# TRAINER = SklearnClassifier(BernoulliNB())
classifier = train_or_load("nb_classifier.pickle", TRAINER, training_set, False)
print(classifier.show_most_informative_features(15))

#classification
test_set = nltk.classify.apply_features(FEATURE_EXTRACTOR.extract_features, test_tweets)
print(nltk.classify.accuracy(classifier, test_set))