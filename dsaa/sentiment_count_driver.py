import os

import settings
from dsaa import text_loader
from sentiment_analysis.preprocessing import PreProcessing


def load_afinn_lexicon_to_map():
    dir = "{}/dsaa/AFINN-111.txt".format(settings.PROJECT_ROOT)

    afinn_map = {}

    with open(dir, "r") as file:
        for line in file.readlines():
            tokens = line.split()
            string = " ".join(tokens[:len(tokens) - 1])
            afinn_map[string] = int(tokens[len(tokens)-1])

    return afinn_map


def determine_polarity(word, lexicon):

    if word not in lexicon:
        return 0

    if lexicon[word] < 0:
        return -1

    if lexicon[word] > 0:
        return 1

    return 0


def get_word_list(docs):
    word_list = []
    for doc in docs:
        word_list.extend(doc.split())
    return word_list


def get_unique_word_list(docs):
    return list(set(get_word_list(docs)))


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
                             PreProcessing.RemoveHashtagSymbol(),
                             PreProcessing.ConcatWordArray()]

            docs = PreProcessing.preprocess_strings(docs, preprocessors)

            unique_word_list = get_unique_word_list(docs)

            afinn_lexicon = load_afinn_lexicon_to_map()
            polarities = [determine_polarity(word, afinn_lexicon )
                          for word in unique_word_list]

            positive = list(filter(lambda x: x > 0, polarities))
            negative = list(filter(lambda x: x < 0, polarities))
            neutral = list(filter(lambda x: x == 0, polarities))

            print("Community {}".format(community_num))
            print("Positive: {}".format(len(positive)))
            print("Negative: {}".format(len(negative)))
            print("Neutral: {}".format(len(neutral)))
            print()
