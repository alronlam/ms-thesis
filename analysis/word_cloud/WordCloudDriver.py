import numpy
import path
from PIL import Image
from gensim import corpora, models
from matplotlib import pyplot
from nltk import RegexpTokenizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from analysis.tf_idf.TfIdfExtractor import generate_tfidf_corpus_dictionary
from community_detection import Utils
from community_detection.Utils import get_vertex_ids_in_each_community, get_user_ids_from_vertex_ids, \
    get_tweet_texts_belonging_to_user_ids, get_vertex_ids_in_each_community_optimized, get_texts_per_community
from sentiment_analysis.preprocessing.PreProcessing import preprocess_strings


def generate_word_cloud_per_community(graph, membership, tweet_objects, base_file_name, preprocessors=[]):
    texts_per_community = get_texts_per_community(graph, membership, tweet_objects, preprocessors)
    texts_per_community = [" ".join(texts) for texts in texts_per_community] # convert to list of strings

    brexit_coloring = numpy.array(Image.open("C:/Users/user/PycharmProjects/ms-thesis/uk_flag.png"))

    stopwords = set(STOPWORDS)
    stopwords.update(Utils.load_function_words("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/preprocessing/eng-function-words.txt"))
    stopwords.update(Utils.load_function_words("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/preprocessing/fil-function-words.txt"))

    word_cloud = WordCloud(background_color="white", max_words=500, mask=brexit_coloring, stopwords=stopwords, max_font_size=40, random_state=42)

    for index, text in enumerate(texts_per_community):
        if text.strip():
            word_cloud.generate(text)
            image_colors = ImageColorGenerator(brexit_coloring)

            pyplot.imshow(word_cloud.recolor(color_func=image_colors), interpolation="bilinear")
            pyplot.axis("off")
            pyplot.figure()
            word_cloud.to_file("word_clouds/{}-community-{}.png".format(base_file_name, index))



def generate_tfidf_word_cloud_per_community(graph, membership, tweet_objects, base_file_name, preprocessors=[]):
    texts_per_community = get_texts_per_community(graph, membership, tweet_objects, preprocessors)
    texts_per_community = [" ".join(texts) for texts in texts_per_community] # convert to list of strings
    tokens_per_community = [text.split() for text in texts_per_community]
    tfidf_model, corpus, dictionary = generate_tfidf_corpus_dictionary(tokens_per_community)

    # brexit_coloring = numpy.array(Image.open("C:/Users/user/PycharmProjects/ms-thesis/uk_flag.png"))
    pdebates_coloring = numpy.array(Image.open("C:/Users/user/PycharmProjects/ms-thesis/ph_flag.png"))



    for index, text in enumerate(texts_per_community):
        print("TF-IDF word cloud: {}/{}".format(index, len(texts_per_community)))

        if text.strip():
            top_words = numpy.sort(numpy.array(tfidf_model[corpus[index]],dtype = [('word',int), ('score',float)]),order='score')[::-1]
            word_cloud = WordCloud(background_color="white", max_words=100, mask=pdebates_coloring, max_font_size=40, random_state=42)
            words_to_fit = [(dictionary[word], score) for (word, score) in top_words]
            word_cloud = word_cloud.fit_words(dict(words_to_fit))
            image_colors = ImageColorGenerator(pdebates_coloring)

            pyplot.imshow(word_cloud.recolor(color_func=image_colors), interpolation="bilinear")
            pyplot.axis("off")
            pyplot.figure()
            word_cloud.to_file("word_clouds/{}-community-{}.png".format(base_file_name, index))
            pyplot.close('all')
