import numpy
from gensim import corpora, models
from nltk import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import STOPWORDS

from community_detection import Utils
from community_detection.Utils import get_vertex_ids_in_each_community, get_user_ids_from_vertex_ids, \
    get_tweet_texts_belonging_to_user_ids
from sentiment_analysis.preprocessing.PreProcessing import preprocess_strings
from twitter_data.database import DBUtils

def generate_tfidf_corpus_dictionary(documents):
    stop_words = set(STOPWORDS)
    stop_words.update(Utils.load_function_words("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/preprocessing/eng-function-words.txt"))
    stop_words.update(Utils.load_function_words("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/preprocessing/fil-function-words.txt"))

    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(document) for document in documents]

    # Run tf-idf
    tfidf = models.TfidfModel(corpus)
    return tfidf, corpus, dictionary

def get_top_keywords_from_documents(documents, TOP_N_KEYWORDS=20):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    try:
        tfidf_vectorizer.fit_transform(documents)
        indices = numpy.argsort(tfidf_vectorizer.idf_)[::-1]
        features = tfidf_vectorizer.get_feature_names()
        top_features = [features[i] for i in indices[:TOP_N_KEYWORDS]]
        top_features = [feature.strip() for feature in top_features if feature.strip()]
        return top_features
    except Exception as e: # probably trying to vectorize empty list
        print("get_top_keywords_from_documents exception: {}".format(e))
        return ""


def get_top_keywords_per_community(graph, membership, tweet_objects, preprocessors=[]):

    keyword_list = []
    vertex_ids_per_community = get_vertex_ids_in_each_community(graph, membership)

    for community_num, vertex_ids in enumerate(vertex_ids_per_community):
        user_ids_str = get_user_ids_from_vertex_ids(graph, vertex_ids)
        tweet_texts = get_tweet_texts_belonging_to_user_ids(tweet_objects, user_ids_str)
        tweet_texts = preprocess_strings(tweet_texts, preprocessors)
        keyword_list.append((community_num, get_top_keywords_from_documents(tweet_texts)))

    return keyword_list


# from sentiment_analysis.preprocessing.PreProcessing import *
#
# import pickle
# graph = pickle.load(open("C:/Users/user/PycharmProjects/ms-thesis/community_detection/100-threshold-0.04-brexit_mention_hashtag_contextualsa_graph.pickle", "rb"))
# membership = pickle.load(open("C:/Users/user/PycharmProjects/ms-thesis/community_detection/100-threshold-0.04-brexit_mention_hashtag_contextualsa_graph.pickle.membership", "rb"))
#
# json_tweet_ids = Utils.load_tweet_ids_from_json_files("D:/DLSU/Masters/MS Thesis/data-2016/test")[:500]
# json_tweet_objects = DBUtils.retrieve_all_tweet_objects_from_db(json_tweet_ids, verbose=True)
#
# brexit_topic_modelling_preprocessors = [SplitWordByWhitespace(),
#                  WordToLowercase(),
#                  ReplaceURL(),
#                  RemovePunctuationFromWords(),
#                  ReplaceUsernameMention(),
#                  RemoveRT(),
#                  RemoveLetterRepetitions(),
#                  RemoveTerm("#brexit"),
#                  RemoveTerm("<url>"),
#                  RemoveTerm("<username>"),
#                  RemoveExactTerms(Utils.load_function_words("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/preprocessing/eng-function-words.txt")),
#                  RemoveExactTerms(Utils.load_function_words("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/preprocessing/fil-function-words.txt")),
#                  ConcatWordArray()]
#
# print(get_top_keywords_per_community(graph, membership, json_tweet_objects, brexit_topic_modelling_preprocessors), file=open("test_tfidf.txt", "w", encoding="utf8"))