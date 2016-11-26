import abc
import pickle
from sentiment_analysis.machine_learning.feature_extraction import FeatureExtractorBase
from sentiment_analysis.preprocessing import PreProcessing

class SubjectivityClassifier(object):

    def preprocess(self, tweet_text):
        for preprocessor in self.preprocessors:
            tweet_text = preprocessor.preprocess_tweet(tweet_text)
        return tweet_text

    @abc.abstractmethod
    def classify_subjectivity(self, tweet_text):
        """
        :param text: string to be analyzed
        :return: "negative" "positive" or "neutral"
        """

class MLSubjectivityClassifier(SubjectivityClassifier):
    def __init__(self, feature_extractor_path, classifier_pickle_path):
        self.feature_extractor = FeatureExtractorBase.load_feature_extractor_from_pickle(feature_extractor_path)
        self.classifier = self.load_classifier_from_pickle(classifier_pickle_path)
        self.preprocessors = [PreProcessing.SplitWordByWhitespace(), PreProcessing.WordLengthFilter(3),
                              PreProcessing.RemovePunctuationFromWords(), PreProcessing.WordToLowercase()]

    def classify_subjectivity(self, tweet_text):
        tweet_text = self.preprocess(tweet_text)
        return self.classifier.classify(self.feature_extractor.extract_features(tweet_text))

    def load_classifier_from_pickle(self, pickle_file_name):
        with open(pickle_file_name, 'rb') as pickle_file:
            return pickle.load(pickle_file)

    def get_name(self):
        return "ML Subjectivity Classifier"


class EmbeddingSubjectivityClassifier(object):

    def load_from_pickle(self, pickle_file_name):
        with open(pickle_file_name, 'rb') as pickle_file:
            return pickle.load(pickle_file)


    @abc.abstractmethod
    def classify_subjectivity(self, tweet_embedding):
        """
        :param tweet_embedding: embedding to be analyzed
        :return: "subjective" or "objective
        """

class MLEmbeddingSubjectivityClassifier(EmbeddingSubjectivityClassifier):



    def __init__(self, classifier_pickle_file_name, scaler_pickle_file_name):
        # self.corpus_w2v = self.load_from_pickle(corpus_pickle_file_name)
        self.classifier = self.load_from_pickle(classifier_pickle_file_name)
        self.scaler = self.load_from_pickle(scaler_pickle_file_name)

    def classify_subjectivity(self, tweet_embedding):
        tweet_embedding = self.scaler.transform(tweet_embedding)
        label = self.classifier.predict(tweet_embedding)
        return label[0]