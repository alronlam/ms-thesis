import abc
import nltk

class FeatureExtractorBase(object):

    def __init__(self, tweet_list):
        self.word_features = self.get_word_features(self.get_words_in_tweets(tweet_list))

    def get_words_in_tweets(tweets):
        all_words = []
        for (words, sentiment) in tweets:
            all_words.extend(words)
        return all_words

    def get_word_features(wordlist):
        wordlist = nltk.FreqDist(wordlist)
        word_features = wordlist.keys()
        return word_features

    @abc.abstractmethod
    def extract_features(self, document):
        document_words = set(document)
        features = {}
        for word in self.word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features