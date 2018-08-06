import copy
import os
import pickle
from collections import Counter

import settings
from dsaa import text_loader
from sentiment_analysis.preprocessing import PreProcessing

OUTPUT_DIR = '{}/dsaa_results/topic_models'.format(settings.PROJECT_ROOT)

preprocessors = [PreProcessing.SplitWordByWhitespace(),
                 PreProcessing.WordToLowercase(),
                 PreProcessing.ReplaceURL(replacement_token="<url>"),
                 PreProcessing.ReplaceUsernameMention(replacement_token="<username>"),
                 PreProcessing.RemoveTerm("<username>"),
                 PreProcessing.RemoveTerm("<url>"),
                 PreProcessing.RemoveExactTerms(["amp", "brexit", "#brexit"],
                                                ignore_case=True),
                 PreProcessing.RemovePunctuationFromWords(),
                 PreProcessing.RemoveRT(),
                 PreProcessing.RemoveLetterRepetitions(),
                 PreProcessing.RemoveNumbers(),
                 PreProcessing.RemoveHashtagSymbol(),
                 PreProcessing.WordLengthFilter(3),
                 PreProcessing.RemoveExactTerms(["amp", "brexit", ],
                                                ignore_case=True),
                 PreProcessing.ConcatWordArray()]


if __name__ == "__main__":

    lda_models = pickle.load(open("{}/scheme_lda_models.pickle".format(OUTPUT_DIR), "rb"))

    for config in text_loader.CONFIGS:
        print("\n\n### {} ###".format(config))
        dir = os.path.join(text_loader.TEXTS_ROOT_DIR, config)

        original_community_docs = text_loader.load_community_docs(dir)
        community_docs = copy.deepcopy(original_community_docs)
        community_docs = [PreProcessing.preprocess_strings(x, preprocessors)
                          for x in community_docs]

        topic_tweets_map = {}
        max_similarity_map = {}

        lda = lda_models[config]

        for community_num, (docs, original_docs) in enumerate(zip(community_docs, original_community_docs)):

            # print("\n***\nCommunity {}".format(community_num))

            for doc, original in zip(docs, original_docs):

                # Build map for most common tweet for each topic
                closest_topic, score = lda.closest_topic(doc, with_score=True)
                curr_list = topic_tweets_map.get(closest_topic, [])
                topic_tweets_map[closest_topic] = curr_list + [original]

                # Build map for most "representative" tweet for each topic
                curr_max_tweet, curr_max_score =max_similarity_map.get(closest_topic, ("", -100))
                if score > curr_max_score:
                    max_similarity_map[closest_topic] = (original, score)

        for topic_num, docs in topic_tweets_map.items():
            print("\n------------\nTopic {}".format(topic_num))
            n = 1
            print("> Most Frequent {} Tweet(s):".format(n))
            counter = Counter(docs)
            most_common = counter.most_common(n)
            most_common_strs = ["{}".format(x[0])
                                for x in most_common]
            collated_str = "\n".join(most_common_strs)
            print(collated_str)

            print("\n> Most Representative Tweet:")
            print("{}".format(max_similarity_map[topic_num][0]))
