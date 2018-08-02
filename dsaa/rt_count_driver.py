import os
from collections import Counter

from dsaa import text_loader
from sentiment_analysis.preprocessing import PreProcessing
from sentiment_analysis.preprocessing.PreProcessing import *


def is_rt(text):
    preprocessors = [SplitWordByWhitespace(),
                     WordToLowercase(),
                     RemovePunctuationFromWords(),
                     ConcatWordArray()]

    preprocessed_text = PreProcessing.preprocess_strings([text], preprocessors)[0]
    tokens = preprocessed_text.split()
    return tokens[0].lower() == "rt"


def most_frequent_tweets(docs, n=3):
    counter = Counter(docs)
    return counter.most_common(n)

if __name__ == "__main__":

    # Read CSV for each community

    for config in text_loader.CONFIGS:
        print("### {} ###".format(config))
        dir = os.path.join(text_loader.TEXTS_ROOT_DIR, config)

        community_docs = text_loader.load_community_docs(dir)

        for community_num, docs in enumerate(community_docs):
            print("Community {}".format(community_num))

            print("All: {}".format(len(docs)))


            unique = set(docs)
            print("Unique: {}".format(len(unique)))

            num_rt = sum([1 for x in unique if is_rt(x)])
            print("Non RT: {}".format(len(unique) - num_rt))

            print("Most Frequent (All):")
            most_frequent = most_frequent_tweets(docs)
            most_frequent_strs = [tweet+"-"+str(count)
                                  for (tweet, count) in most_frequent]
            print("\n".join(most_frequent_strs))
            print()


