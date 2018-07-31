import os

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
                             PreProcessing.ConcatWordArray()]

            docs = PreProcessing.preprocess_strings(docs, preprocessors)

            unique_hashtags = sorted(list(get_unique_hashtags(docs)))

            print("*** Community {} - {} unique hashtags ***".format(community_num, len(unique_hashtags)))
            print("\n".join(list(unique_hashtags)))
            print()