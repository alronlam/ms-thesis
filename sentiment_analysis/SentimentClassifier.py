from sentiment_analysis.classifier.ClassifierIO import *
from sentiment_analysis.feature_extraction import FeatureExtractorBase
from sentiment_analysis.preprocessing import PreProcessing
from database import LexiconManager
import abc

class SentimentClassifier(object):

    def preprocess(self, tweet_text):
        for preprocessor in self.preprocessors:
            tweet_text = preprocessor.preprocess_tweet(tweet_text)
        return tweet_text

    @abc.abstractmethod
    def classify_sentiment(self, text):
        """
        :param text: string to be analyzed
        :return: "negative" "positive" or "neutral"
        """

class MLClassifier(SentimentClassifier):
    def __init__(self):
        self.feature_extractor = FeatureExtractorBase.load_feature_extractor_from_pickle("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/unigram_feature_extractor.pickle")
        self.classifier = load_classifier_from_pickle("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/nb_classifier.pickle")

    def classify_sentiment(self, tweet_text):
        tweet_text = self.preprocess(tweet_text)
        return self.classifier.classify(self.feature_extractor.extract_features(tweet_text))

class LexiconClassifier(SentimentClassifier):

    def __init__(self):
        self.preprocessors = [PreProcessing.SplitWordByWhitespace(), PreProcessing.WordToLowercase(), PreProcessing.RemovePunctuationFromWords()]

    def get_overall_sentiment_score(self, tweet_text):
        tweet_text = self.preprocess(tweet_text)
        tweet_word_sentiment_scores = []

        # get all scores for the words in the text
        for tweet_word in tweet_text:
            tweet_word_sentiment_scores.append(LexiconManager.get_sentiment_score(tweet_word))

        return sum(tweet_word_sentiment_scores)

    def classify_sentiment(self, tweet_text):
        sentiment_score = self.get_overall_sentiment_score(tweet_text)

        if sentiment_score > 0:
            return "positive"
        elif sentiment_score < 0:
            return "negative"
        else:
            return "neutral"
