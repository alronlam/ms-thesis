import numpy
import path
from PIL import Image
from matplotlib import pyplot
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from community_detection import Utils
from community_detection.Utils import get_vertex_ids_in_each_community, get_user_ids_from_vertex_ids, \
    get_tweet_texts_belonging_to_user_ids, get_vertex_ids_in_each_community_optimized
from sentiment_analysis.preprocessing.PreProcessing import preprocess_strings


def get_texts_per_community(graph, membership, tweet_objects, preprocessors=[]):

    texts_per_community = []

    vertex_ids_per_community = get_vertex_ids_in_each_community_optimized(graph, membership)

    for community_num, vertex_ids in enumerate(vertex_ids_per_community):
        user_ids_str = get_user_ids_from_vertex_ids(graph, vertex_ids)
        tweet_texts = get_tweet_texts_belonging_to_user_ids(tweet_objects, user_ids_str)
        tweet_texts = preprocess_strings(tweet_texts, preprocessors)
        texts_per_community.append(tweet_texts)

    return texts_per_community


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
