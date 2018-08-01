import os
from collections import Counter

from dsaa import text_loader
from sentiment_analysis.preprocessing import PreProcessing


def get_hashtags(text):
    tokens = text.split()
    hashtags = [token.lower()
                for token in tokens
                if token[0] == "#" and token.count("#") == 1]
    return hashtags

def get_hashtags_list(text_list):
    hashtags = []
    for text in text_list:
        hashtags.extend(get_hashtags(text))
    return hashtags

def get_unique_hashtags(text_list):
    return set(get_hashtags_list(text_list))


def get_top_n_hashtags(text_list, n=5):
    hashtags = get_hashtags_list(text_list)
    counter = Counter(hashtags)
    return counter.most_common(n)


if __name__ == "__main__":

    # Read CSV for each community

    for config in text_loader.CONFIGS:
        print("### {} ###".format(config))
        dir = os.path.join(text_loader.TEXTS_ROOT_DIR, config)

        community_docs = text_loader.load_community_docs(dir)

        for community_num, docs in enumerate(community_docs):

            preprocessors = [PreProcessing.SplitWordByWhitespace(),
                             PreProcessing.WordToLowercase(),
                             PreProcessing.RemovePunctuationFromWords(),
                             PreProcessing.RemoveTerm('…', ignore_case=True),
                             PreProcessing.RemoveTerm('brexit', ignore_case=True),
                             PreProcessing.RemoveExactTerms(['#brexit'], ignore_case=True),
                             PreProcessing.ConcatWordArray()]

            docs = PreProcessing.preprocess_strings(docs, preprocessors)

            print("*** Community {}".format(community_num))

            unique_hashtags = sorted(list(get_unique_hashtags(docs)))
            print("{} unique hashtags".format(len(unique_hashtags)))
            # print("\n".join(list(unique_hashtags)))

            top_hashtags = get_top_n_hashtags(docs)
            for (hashtag, count) in top_hashtags:
                print("{} - {}".format(hashtag, count))

            print()

