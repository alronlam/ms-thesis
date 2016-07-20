import abc
import string
class PreProcessor(object):

    @abc.abstractmethod
    def preprocess_tweet(self, tweet):
        """
        :return: pre-process a single tweet
        """


class WordLengthFilter(PreProcessor):

    def __init__(self, min_word_length):
        self.min_word_length = min_word_length

    # expects list of words in tweet
    def preprocess_tweet(self, tweet_words):
         return [word for word in tweet_words if len(word) >= self.min_word_length]

class WordToLowercase(PreProcessor):

    # expects list of words in tweet
    def preprocess_tweet(self, tweet_words):
        return [word.lower() for word in tweet_words]

class SplitWordByWhitespace(PreProcessor):

    # expects tweet string
    def preprocess_tweet(self, tweet):
        return tweet.split()

class RemovePunctuationFromWords(PreProcessor):

    def __init__(self):
        self.translator = str.maketrans({key: None for key in string.punctuation})

    def preprocess_tweet(self, tweet_words):
        return [word.translate(self.translator) for word in tweet_words ]


def preprocess_tweets(tweets, preprocessors):
    preprocessed_tweets = []
    for tweet in tweets:
        for preprocessor in preprocessors:
            tweet = preprocessor.preprocess_tweet(tweet)
        preprocessed_tweets.append(tweet)
    return preprocessed_tweets