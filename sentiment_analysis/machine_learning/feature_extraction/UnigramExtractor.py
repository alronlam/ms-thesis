import nltk

from sentiment_analysis.machine_learning.feature_extraction.FeatureExtractorBase import FeatureExtractorBase


class UnigramExtractor(FeatureExtractorBase):

    def __init__(self, labeled_tweets, top_n_keywords = None):
        self.top_n_keywords = top_n_keywords
        self.word_features = self.get_word_features(self.get_words_in_tweets(labeled_tweets))

    @staticmethod
    def get_words_in_tweets(labeled_tweets):
        all_words = []
        for (tweet_words, sentiment) in labeled_tweets:
            all_words.extend(tweet_words)
        return all_words

    def get_word_features(self, wordlist):
        wordlist = nltk.FreqDist(wordlist)
        max_num_keywords = self.top_n_keywords if self.top_n_keywords else wordlist.__len__()
        word_features = [word for word, count in wordlist.most_common(max_num_keywords)]
        return word_features

    def extract_features(self, document):
        document_words = set(document)
        features = {}
        for word in self.word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features
