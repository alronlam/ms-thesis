import copy
import os
import pickle
from collections import Counter

import settings
from dsaa import text_loader, utils
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

    utils.create_path_if_not_exists(OUTPUT_DIR)

    log_file_path = "{}/topic_models_raw_log.txt".format(OUTPUT_DIR)

    lda_models = pickle.load(open("{}/scheme_lda_models.pickle".format(OUTPUT_DIR), "rb"))

    with open(log_file_path, "w") as log_file:
        for config in text_loader.CONFIGS:
            print("### {} ###".format(config), file=log_file)
            log_file.flush()
            dir = os.path.join(text_loader.TEXTS_ROOT_DIR, config)

            community_docs = text_loader.load_community_docs(dir)
            original_community_docs = copy.deepcopy(community_docs)
            community_docs = [PreProcessing.preprocess_strings(x, preprocessors)
                              for x in community_docs]

            closest_topic_map = {}
            lda = lda_models[config]

            for community_num, (docs, original_docs) in enumerate(zip(community_docs, original_community_docs)):

                print("\n***\nCommunity {}".format(community_num))

                for doc, original in zip(docs, original_docs):
                    closest_topic = lda.closest_topic(doc)
                    curr_list = closest_topic_map.get(closest_topic, [])
                    closest_topic_map[closest_topic] = curr_list + [original]

                for topic_num, docs in closest_topic_map.items():
                    print("\nTopic {}".format(topic_num))
                    counter = Counter(docs)
                    most_common = counter.most_common(3)
                    most_common_strs = ["({}) {}".format(x[1], x[0])
                                        for x in most_common]
                    collated_str = "\n".join(most_common_strs)
                    print(collated_str)
