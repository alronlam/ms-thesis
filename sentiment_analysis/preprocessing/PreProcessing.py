import abc
class PreProcessor(object):

    @abc.abstractmethod
    def preprocess_tweet(self, tweet):
        """
        :return: pre-process a single tweet
        """


class WordLengthFilter(PreProcessor):

    def __init__(self, min_word_length):
        self.min_word_length = min_word_length

    def preprocess_tweet(self, tweet_words):
         return [word for word in tweet_words if len(word) >= self.min_word_length]

class WordToLowercase(PreProcessor):
    def preprocess_tweet(self, tweet_words):
        return [word.lower() for word in tweet_words]

def preprocess_tweets(labeled_tweets, preprocessors):
    preprocessed_tweets = []
    for tweet_words, label in labeled_tweets:
        for preprocessor in preprocessors:
            processed_tweet_words = preprocessor.preprocess_tweet(tweet_words)
        preprocessed_tweets.append((processed_tweet_words, label))
    return preprocessed_tweets