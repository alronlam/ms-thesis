from matplotlib import pyplot
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def generate_word_cloud(texts, output_file, max_words=30, mask=None, colors=None):
    concatenated_texts = " ".join(texts)

    stopwords = set(STOPWORDS)

    word_cloud = WordCloud(background_color="white",
                           max_words=max_words,
                           mask=mask,
                           stopwords=stopwords,
                           max_font_size=40,
                           random_state=42,
                           collocations=False)

    word_cloud.generate(concatenated_texts)

    if colors is not None:
        image_colors = ImageColorGenerator(colors)
        pyplot.imshow(word_cloud.recolor(color_func=image_colors), interpolation="bilinear")
    else:
        pyplot.imshow(word_cloud, interpolation="bilinear")

    pyplot.axis("off")
    pyplot.figure()
    word_cloud.to_file(output_file)
