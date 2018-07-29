import os

import bokeh.plotting as bp
import lda
import numpy as np
from bokeh.io import save
from bokeh.models import HoverTool
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

from scratch import load_community_docs
from sentiment_analysis.preprocessing import PreProcessing
from sentiment_analysis.preprocessing.PreProcessing import *

### CONSTANTS ###


n_top_words = 10  # number of keywords we show

# 20 colors
colormap = np.array([
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
])

lda_preprocessors = [SplitWordByWhitespace(),
                     WordToLowercase(),
                     ReplaceURL(),
                     RemoveTerm("<url>"),
                     RemoveTerm("http"),
                     ReplaceUsernameMention(),
                     RemoveTerm("<username>"),
                     RemoveTerm("#"),
                     RemovePunctuationFromWords(),
                     RemoveRT(),
                     RemoveLetterRepetitions(),
                     # Stem(),
                     WordLengthFilter(3),
                     RemoveExactTerms(["amp"]),
                     ConcatWordArray()
                     ]


def viz(docs, lda_model, X_topics, cvectorizer, tsne_lda, title):

    ### Generate topic summaries ###
    topic_summaries = []
    topic_word = lda_model.topic_word_  # all topic words
    vocab = cvectorizer.get_feature_names()
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]  # get!
        topic_summaries.append(' '.join(topic_words))  # append!


    _lda_keys = []
    for i in range(X_topics.shape[0]):
        _lda_keys += X_topics[i].argmax(),

    colors = [colormap[i] for i in _lda_keys]
    groups = [topic_summaries[i] for i in _lda_keys]

    plot_lda = bp.figure(plot_width=1080, plot_height=720,
                         title=title,
                         tools="pan,wheel_zoom,reset,hover,previewsave",
                         x_axis_type=None, y_axis_type=None, min_border=1)

    source = bp.ColumnDataSource(data=dict(x=tsne_lda[:, 0],
                                           y=tsne_lda[:, 1],
                                           content=docs,
                                           color=colors,
                                           legend=groups))

    plot_lda.scatter(x='x', y='y', color='color', legend='legend', source=source)

    # # randomly choose a news (within a topic) coordinate as the crucial words coordinate
    # topic_coord = np.empty((X_topics.shape[1], 2)) * np.nan
    # for topic_num in _lda_keys:
    #     if not np.isnan(topic_coord).any():
    #         break
    #     topic_coord[topic_num] = tsne_lda[_lda_keys.index(topic_num)]
    #
    # # plot crucial words
    # for i in range(X_topics.shape[1]):
    #     plot_lda.text(topic_coord[i, 0], topic_coord[i, 1], [topic_summaries[i]])

    # hover tools
    hover = plot_lda.select(dict(type=HoverTool))
    hover.tooltips = {"tweet": "@content"}

    # save the plot
    folder = 'dsaa_results/tsne/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    save(plot_lda, 'dsaa_results/tsne/{}.html'.format(title))


### MAIN ###

def main():
    configs = [
        "mentions",
        "hashtags",
        "sa",
        "contextualsa",
        "scoring"
    ]

    # Read CSV for each community

    for config in configs:
        print("### {} ###".format(config))

        folder = "dsaa_results/texts/{}/".format(config)

        community_docs = load_community_docs(folder)
        for community_num, orig_docs in enumerate(community_docs):
            docs = PreProcessing.preprocess_strings(orig_docs, lda_preprocessors)


            ### LDA ###

            n_topics = 5  # number of topics
            n_iter = 500  # number of iterations

            # vectorizer: ignore English stopwords & words that occur less than 5 times
            cvectorizer = CountVectorizer(stop_words='english')
            cvz = cvectorizer.fit_transform(docs)

            # train an LDA model
            lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter)
            X_topics = lda_model.fit_transform(cvz)

            threshold = 0.5
            _idx = np.amax(X_topics, axis=1) > threshold  # idx of doc that above the threshold
            X_topics = X_topics[_idx]

            ### TSNE ###

            # a t-SNE model
            # angle value close to 1 means sacrificing accuracy for speed
            # pca initializtion usually leads to better results
            tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')

            # 20-D -> 2-D
            tsne_lda = tsne_model.fit_transform(X_topics)

            ### VIZ ###
            title = "{}-{}".format(config, community_num)
            viz(orig_docs, lda_model, X_topics, cvectorizer, tsne_lda, title)


main()
