import abc
import pickle
import numpy

from sklearn.preprocessing import scale
from afinn import Afinn

from sentiment_analysis.lexicon.simple.database import LexiconManager
from sentiment_analysis.lexicon.anew.database import ANEWLexiconManager
from sentiment_analysis.machine_learning.feature_extraction import FeatureExtractorBase
from sentiment_analysis.preprocessing import PreProcessing

from sentiment_analysis.subjectivity import SubjectivityClassifier


class SentimentClassifier(object):
    def preprocess(self, tweet_text):
        for preprocessor in self.preprocessors:
            tweet_text = preprocessor.preprocess_tweet(tweet_text)
        return tweet_text

    def get_sum_based_score(self, tweet_text, lexicon_manager):
        tweet_text = self.preprocess(tweet_text)
        tweet_word_sentiment_scores = []

        # get all scores for the words in the text
        for tweet_word in tweet_text:
            sentiment_score = lexicon_manager.get_sentiment_score(tweet_word)
            if sentiment_score < 0:
                sentiment_score *= 1.8
            tweet_word_sentiment_scores.append(sentiment_score)
        return sum(tweet_word_sentiment_scores)

    def get_majority_score(self, tweet_text, lexicon_manager):
        tweet_text = self.preprocess(tweet_text)
        positive_count = 0
        negative_count = 0

        for tweet_word in tweet_text:
            sentiment_score = lexicon_manager.get_sentiment_score(tweet_word)
            if sentiment_score > 0:
                positive_count += 1
            elif sentiment_score < 0:
                negative_count += 1

        return positive_count + negative_count

    def load_from_pickle(self, pickle_file_name):
        with open(pickle_file_name, 'rb') as pickle_file:
            return pickle.load(pickle_file)

    @abc.abstractmethod
    def classify_sentiment(self, tweet_text, contextual_info_dict):
        """
        :param tweet_text: string to be analyzed
        :param contextual_info_dict: dictionary containing any additional info available
        :return: "negative" "positive" or "neutral"
        """

    @abc.abstractmethod
    def get_name(self):
        """
        :return: short name describing the classifier
        """

from gensim.models.word2vec import Word2Vec
class ConversationalContextClassifier(SentimentClassifier):

    def __init__(self, corpus_bin_file_name, classifier_pickle_file_name, scaler_pickle_file_name):
        self.preprocessors = [PreProcessing.SplitWordByWhitespace(), PreProcessing.WordToLowercase(),
                     PreProcessing.RemovePunctuationFromWords()]
        self.corpus_w2v = Word2Vec.load_word2vec_format(corpus_bin_file_name, binary=True)
        # self.corpus_w2v = self.load_from_pickle(corpus_pickle_file_name)
        self.classifier = self.load_from_pickle(classifier_pickle_file_name)
        self.scaler = self.load_from_pickle(scaler_pickle_file_name)

    def classify_sentiment(self, tweet_text, contextual_info_dict):
        tweet_text = self.preprocess(tweet_text)
        text_vector = self.buildWordVector(tweet_text, 300)
        text_vector = self.scaler.transform(text_vector)
        label = self.classifier.predict(text_vector)
        return label[0]

    def buildWordVector(self, text, size):
        vec = numpy.zeros(size).reshape((1, size))
        count = 0.
        for word in text:
            try:
                vec += self.corpus_w2v[word].reshape((1, size))
                count += 1.
            except KeyError:
                continue
        if count != 0:
            vec /= count
        return vec

    def get_name(self):
        return "word2vec-svm"

class MLClassifier(SentimentClassifier):
    def __init__(self, feature_extractor_path, classifier_pickle_path):
        self.feature_extractor = FeatureExtractorBase.load_feature_extractor_from_pickle(feature_extractor_path)
        self.classifier = self.load_classifier_from_pickle(classifier_pickle_path)
        self.preprocessors = [PreProcessing.SplitWordByWhitespace(), PreProcessing.WordToLowercase(),
                              PreProcessing.RemovePunctuationFromWords()]

    def classify_sentiment(self, tweet_text, contextual_info_dict):
        tweet_text = self.preprocess(tweet_text)
        return self.classifier.classify(self.feature_extractor.extract_features(tweet_text))

    def load_classifier_from_pickle(self, pickle_file_name):
        with open(pickle_file_name, 'rb') as pickle_file:
            return pickle.load(pickle_file)

    def get_name(self):
        return "ML"


class WiebeLexiconClassifier(SentimentClassifier):
    def __init__(self):
        self.preprocessors = [PreProcessing.SplitWordByWhitespace(), PreProcessing.WordToLowercase(),
                              PreProcessing.RemovePunctuationFromWords()]

    def get_overall_sentiment_score(self, tweet_text):
        return self.get_majority_score(tweet_text, LexiconManager)
        # return self.get_sum_based_score(tweet_text, LexiconManager)

    def classify_sentiment(self, tweet_text, contextual_info_dict):
        sentiment_score = self.get_overall_sentiment_score(tweet_text)

        if sentiment_score > 0:
            return "positive"
        elif sentiment_score < 0:
            return "negative"
        else:
            return "neutral"

    def get_name(self):
        return "Lexicon_Wiebe"


class ANEWLexiconClassifier(SentimentClassifier):
    def __init__(self):
        self.preprocessors = [PreProcessing.SplitWordByWhitespace(), PreProcessing.WordToLowercase(),
                              PreProcessing.RemovePunctuationFromWords()]

    def get_overall_sentiment_score(self, tweet_text):
        tweet_text = self.preprocess(tweet_text)
        tweet_word_sentiment_scores = []

        # get all scores for the words in the text
        for tweet_word in tweet_text:
            sentiment_score = ANEWLexiconManager.get_sentiment_score((tweet_word))
            if sentiment_score < 0:
                sentiment_score *= 1.8
            tweet_word_sentiment_scores.append(sentiment_score)

        return sum(tweet_word_sentiment_scores)

    def classify_sentiment(self, tweet_text, contextual_info_dict):
        sentiment_score = self.get_overall_sentiment_score(tweet_text)

        if sentiment_score > 0.5:
            return "positive"
        elif sentiment_score < -0.5:
            return "negative"
        else:
            return "neutral"

    def get_name(self):
        return "Lexicon_ANEW"


class AFINNLexiconClassifier(SentimentClassifier):
    def __init__(self):
        self.preprocessors = [PreProcessing.SplitWordByWhitespace(), PreProcessing.WordToLowercase(),
                              PreProcessing.RemovePunctuationFromWords()]
        self.afinn = Afinn()

    def get_overall_sentiment_score(self, tweet_text):
        tweet_text = self.preprocess(tweet_text)
        return self.afinn.score(" ".join(tweet_text))

    def classify_sentiment(self, tweet_text, contextual_info_dict):
        sentiment_score = self.get_overall_sentiment_score(tweet_text)

        if sentiment_score > 0.5:
            return "positive"
        elif sentiment_score < -0.5:
            return "negative"
        else:
            return "neutral"

    def get_name(self):
        return "Lexicon_AFINN"
