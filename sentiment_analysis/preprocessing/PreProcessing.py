import abc
import copy
import string

import re

class PreProcessor(object):

    @abc.abstractmethod
    def preprocess_text(self, tweet):
        """
        :return: pre-process a single tweet
        """


class WordLengthFilter(PreProcessor):

    def __init__(self, min_word_length):
        self.min_word_length = min_word_length

    # expects list of words in tweet
    def preprocess_text(self, tweet_words):
         return [word.strip() for word in tweet_words if len(word.strip()) >= self.min_word_length]

class WordToLowercase(PreProcessor):

    # expects list of words in tweet
    def preprocess_text(self, tweet_words):
        return [word.lower() for word in tweet_words]

class SplitWordByWhitespace(PreProcessor):

    # expects tweet string
    def preprocess_text(self, tweet):
        return tweet.split()

class RemovePunctuationFromWords(PreProcessor):

    def __init__(self):
        # Do not remove the special tokens (# - hashtag, @ - username, <> - URL/Username replacement)
        self.translator = str.maketrans({key: " " for key in string.punctuation if key != "#" and key != "@" and key !="<" and key != ">"})

    def preprocess_text(self, tweet_words):
        removed_punctuations = [word.translate(self.translator) for word in tweet_words ]
        new_arr = []
        for word in removed_punctuations:
            new_arr.extend(word.split())
        return new_arr

class ReplaceUsernameMention(PreProcessor):

    def __init__(self, replacement_token="<USERNAME>"):
        self.replacement_token = replacement_token
        self.regex = re.compile(r"@[\S]+")

    def preprocess_text(self, text_words):
        return [self.regex.sub(self.replacement_token, word) for word in text_words]

class ReplaceURL(PreProcessor):

    def __init__(self, replacement_token="<URL>"):
        self.replacement_token = replacement_token
        self.regex = re.compile(r"https?:\/\/\S*")

    def preprocess_text(self, text_words):
        return [self.regex.sub(self.replacement_token, word) for word in text_words]

class RemoveRT(PreProcessor):

    def __init__(self, replacement_token=""):
        self.replacement_token = replacement_token
        self.regex = re.compile(r"\brt\b|\bRT\b")

    def preprocess_text(self, text_words):
        return [word for word in text_words if not self.regex.match(word)]

class RemoveLetterRepetitions(PreProcessor):

    def preprocess_text(self, text_words):
        return [re.sub(r"([a-z])\1\1+", r"\1", word) for word in text_words]

class ConcatWordArray(PreProcessor):

    def preprocess_text(self, text_words):
        return " ".join(text_words)

class RemoveTerm(PreProcessor):

    def __init__(self, term, ignore_case=True):
        self.ignore_case=ignore_case
        if self.ignore_case:
            self.term = term.lower()

    def preprocess_text(self, text_words):
        new_array = []
        for word in text_words:
            if self.ignore_case:
                to_compare = word.lower()
            else:
                to_compare = word

            if self.term not in to_compare:
                new_array.append(word)
        return new_array

class RemoveExactTerms(PreProcessor):
    def __init__(self, terms, ignore_case=True):
        self.ignore_case = ignore_case
        if self.ignore_case:
            self.terms = [term.lower() for term in terms]

    def preprocess_text(self, text_words):
        new_array = []
        for word in text_words:
            if self.ignore_case:
                word = word.lower()
            if word not in self.terms:
                new_array.append(word)
        return new_array

def preprocess_strings(strings, preprocessors):
    preprocessed_tweets = []
    for tweet in strings:
        for preprocessor in preprocessors:
            tweet = preprocessor.preprocess_text(tweet)
        preprocessed_tweets.append(tweet)
    return preprocessed_tweets

def preprocess_tweets(tweets, preprocessors):
    tweets_copy = copy.deepcopy(tweets)
    for tweet in tweets_copy:
        for preprocessor in preprocessors:
            tweet.text = preprocessor.preprocess_text(tweet.text)
    return tweets_copy

