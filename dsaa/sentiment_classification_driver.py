import os
from collections import Counter

from dsaa import text_loader
from sentiment_analysis import SentimentClassifier
from sentiment_analysis.preprocessing import PreProcessing

afinn_classifier = SentimentClassifier.AFINNLexiconClassifier()


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

            sentiments = [afinn_classifier.classify_sentiment(x, {})
                          for x in docs]


            counter = Counter(sentiments)


            print("Community {}".format(community_num))
            print(counter['positive'])
            print(counter['negative'])
            print(counter['neutral'])