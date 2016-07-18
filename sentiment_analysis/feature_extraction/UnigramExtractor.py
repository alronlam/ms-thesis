import nltk
from sentiment_analysis.feature_extraction.FeatureExtractorBase import FeatureExtractorBase

class UnigramExtractor(FeatureExtractorBase):

    def __init__(self, tweet_list):
        self.word_features = self.get_word_features(self.get_words_in_tweets(tweet_list))

    @staticmethod
    def get_words_in_tweets(tweets):
        all_words = []
        for (words, sentiment) in tweets:
            all_words.extend(words)
        return all_words

    @staticmethod
    def get_word_features(wordlist):
        wordlist = nltk.FreqDist(wordlist)
        word_features = wordlist.keys()
        return word_features

    def extract_features(self, document):
        document_words = set(document)
        features = {}
        for word in self.word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features
