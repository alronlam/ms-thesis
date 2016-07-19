
from sentiment_analysis.feature_extraction.UnigramExtractor import UnigramExtractor
from sentiment_analysis.classifier.ClassifierIO import *
from sentiment_analysis.feature_extraction import FeatureExtractorBase

class SAFacade(object):
    def __init__(self):
        self.feature_extractor = FeatureExtractorBase.load_feature_extractor_from_pickle("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/unigram_feature_extractor.pickle")
        self.classifier = load_classifier_from_pickle("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/nb_classifier.pickle")

    def classify_sentiment(self, text):
        return self.classifier.classify(self.feature_extractor.extract_features(text.split()))