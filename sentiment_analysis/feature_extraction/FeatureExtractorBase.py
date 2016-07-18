import abc
import nltk

class FeatureExtractorBase(object):

    @abc.abstractmethod
    def extract_features(self):
        """
        :param document: extract feature from this document
        :return: features
        """