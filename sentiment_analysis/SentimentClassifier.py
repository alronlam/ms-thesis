import abc
import pickle
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import scale
from afinn import Afinn

from sentiment_analysis.lexicon.simple.database import LexiconManager
from sentiment_analysis.lexicon.anew.database import ANEWLexiconManager
from sentiment_analysis.machine_learning.feature_extraction import FeatureExtractorBase
from sentiment_analysis.preprocessing import PreProcessing

# from gensim.models.word2vec import Word2Vec
from sentiment_analysis.subjectivity import SubjectivityClassifier


class SentimentClassifier(object):
    def preprocess(self, tweet_text):
        for preprocessor in self.preprocessors:
            tweet_text = preprocessor.preprocess_text(tweet_text)
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


# class ConversationalContextClassifier(SentimentClassifier):
#
#     def __init__(self, corpus_bin_file_name, classifier_pickle_file_name, scaler_pickle_file_name):
#         self.preprocessors = [PreProcessing.SplitWordByWhitespace(), PreProcessing.WordToLowercase(),
#                      PreProcessing.RemovePunctuationFromWords()]
#         self.corpus_w2v = Word2Vec.load_word2vec_format(corpus_bin_file_name, binary=True)
#         # self.corpus_w2v = self.load_from_pickle(corpus_pickle_file_name)
#         self.classifier = self.load_from_pickle(classifier_pickle_file_name)
#         self.scaler = self.load_from_pickle(scaler_pickle_file_name)
#
#     def classify_sentiment(self, tweet_text, contextual_info_dict):
#         tweet_text = self.preprocess(tweet_text)
#         text_vector = self.buildWordVector(tweet_text, 300)
#         text_vector = self.scaler.transform(text_vector)
#         label = self.classifier.predict(text_vector)
#         return label[0]
#
#     def buildWordVector(self, text, size):
#         vec = numpy.zeros(size).reshape((1, size))
#         count = 0.
#         for word in text:
#             try:
#                 vec += self.corpus_w2v[word].reshape((1, size))
#                 count += 1.
#             except KeyError:
#                 continue
#         if count != 0:
#             vec /= count
#         return vec
#
#     def get_name(self):
#         return "word2vec-svm"

class MLClassifier(SentimentClassifier):
    def __init__(self, feature_extractor_path, classifier_pickle_path):
        self.feature_extractor = FeatureExtractorBase.load_feature_extractor_from_pickle(feature_extractor_path)
        self.classifier = pickle.load(open(classifier_pickle_path, 'rb'))
        self.preprocessors = [PreProcessing.SplitWordByWhitespace(), PreProcessing.WordToLowercase(),
                              PreProcessing.RemovePunctuationFromWords()]

    def classify_sentiment(self, tweet_text, contextual_info_dict):
        tweet_text = self.preprocess(tweet_text)
        return self.classifier.classify(self.feature_extractor.extract_features(tweet_text))

    def get_name(self):
        return "ML"




class KerasClassifier(SentimentClassifier):

    MAX_SEQUENCE_LENGTH = 32
    CATEGORIES = ["negative", "neutral", "positive"]

    def __init__(self, tokenizer_pickle_path, classifier_json_path, classifier_weights_path, with_context=False):
        from keras.models import model_from_json
        self.tokenizer = pickle.load(open(tokenizer_pickle_path, "rb"))
        self.classifier = model_from_json([line for line in open(classifier_json_path, "r")][0])
        self.classifier.load_weights(classifier_weights_path)
        self.preprocessors = [PreProcessing.SplitWordByWhitespace(), PreProcessing.WordToLowercase(),
                              PreProcessing.RemovePunctuationFromWords(), PreProcessing.ConcatWordArray()]
        self.with_context = with_context

    def convert_to_word_sequence(self, text):
        from keras.preprocessing.sequence import pad_sequences
        text_arr = [text]
        return pad_sequences(self.tokenizer.texts_to_sequences(text_arr), maxlen=self.MAX_SEQUENCE_LENGTH)

    def convert_numerical_category_to_word(self, number):
        return self.CATEGORIES[number]


    def convert_contextual_tweets_by_idf(self, contextual_tweets, TOP_N_KEYWORDS=32):

        top_keywords = []

        for curr_contextual_tweets in contextual_tweets:
            try:
                tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
                tfidf_vectorizer.fit_transform(curr_contextual_tweets)

                indices = numpy.argsort(tfidf_vectorizer.idf_)[::-1]
                features = tfidf_vectorizer.get_feature_names()
                top_features = [features[i] for i in indices[:TOP_N_KEYWORDS]]
                print(top_features)
            except Exception as e:
                print(e)
                top_features = []

            top_feature_string = " ".join(top_features)
            top_keywords.append(top_feature_string)

        top_keywords = [keyword.strip() for keyword in top_keywords if keyword.strip()]

        return top_keywords

    def convert_contextual_tweets_to_word_sequence(self, contextual_tweets):
        top_keywords = self.convert_contextual_tweets_by_idf(contextual_tweets)
        from keras.preprocessing.sequence import pad_sequences
        return numpy.reshape(pad_sequences([self.tokenizer.texts_to_sequences(top_keywords)], maxlen=self.MAX_SEQUENCE_LENGTH), (1,32))

    def classify_sentiment(self, tweet_text, contextual_info_dict):
        tweet_text = self.preprocess(tweet_text)
        tweet_text_sequence = self.convert_to_word_sequence(tweet_text)
        conv_text_sequence = self.convert_contextual_tweets_to_word_sequence(contextual_info_dict["conv_context"])

        if self.with_context:
            prediction_probabilities = self.classifier.predict([tweet_text_sequence, conv_text_sequence], batch_size=1, verbose=0)
        else:
            prediction_probabilities = self.classifier.predict(tweet_text_sequence,batch_size=1, verbose=0)

        prediction = prediction_probabilities.argmax(axis=1)[0]
        # print("{}\n{}\n\n".format(tweet_text, self.convert_numerical_category_to_word(prediction)))
        return self.convert_numerical_category_to_word(prediction)


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
