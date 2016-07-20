import abc
import nltk
import pickle

class FeatureExtractorBase(object):

    @abc.abstractmethod
    def extract_features(self):
        """
        :param document: extract feature from this document
        :return: features
        """


def save_feature_extractor(file_name, feature_extractor):
    f = open(file_name, 'wb+')
    pickle.dump(feature_extractor, f)
    f.close()


def load_feature_extractor_from_pickle(pickle_file_name):
    f = open(pickle_file_name, 'rb')
    feature_extractor = pickle.load(f)
    f.close()
    return feature_extractor
