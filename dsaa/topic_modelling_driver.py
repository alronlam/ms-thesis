import os
from collections import Counter

import numpy

import settings
from analysis.topic_modelling.LDATopicModeller import LDATopicModeller
from analysis.topic_modelling.TopicModelSimilarityDriver import generate_community_pairwise_similarity_matrix
from dsaa import text_loader, utils
from sentiment_analysis.preprocessing import PreProcessing

OUTPUT_DIR = '{}/dsaa_results/word_clouds/'.format(settings.PROJECT_ROOT)

preprocessors = [PreProcessing.SplitWordByWhitespace(),
                 PreProcessing.WordToLowercase(),
                 PreProcessing.ReplaceURL(replacement_token="<url>"),
                 PreProcessing.RemoveTerm("<url>"),
                 PreProcessing.RemoveExactTerms(["#brexit", "amp"]),
                 PreProcessing.RemovePunctuationFromWords(),
                 PreProcessing.RemoveRT(),
                 PreProcessing.RemoveLetterRepetitions(),
                 PreProcessing.RemoveNumbers(),
                 PreProcessing.RemoveHashtagSymbol(),
                 PreProcessing.WordLengthFilter(3),
                 PreProcessing.RemoveExactTerms(["amp"]),
                 PreProcessing.ConcatWordArray()]

if __name__ == "__main__":

    log_file_path = "{}/dsaa_results/topic_models_raw_log.txt".format(settings.PROJECT_ROOT)
    with open(log_file_path, "w") as log_file:

        utils.create_path_if_not_exists(OUTPUT_DIR)

        # Read CSV for each community

        lda_models = {}

        for config in text_loader.CONFIGS:
            print("### {} ###".format(config), file=log_file)
            log_file.flush()
            dir = os.path.join(text_loader.TEXTS_ROOT_DIR, config)

            community_docs = text_loader.load_community_docs(dir)
            community_docs = [PreProcessing.preprocess_strings(x, preprocessors)
                              for x in community_docs]

            ### Generate Topic Models as a whole ###
            scheme_docs = []
            for docs in community_docs:
                scheme_docs.extend(docs)

            lda = LDATopicModeller(num_topics=5)
            lda.generate_topic_models(scheme_docs)
            lda_models[config] = lda

            print(lda.generate_topic_model_string_from_tuples(), file=log_file)
            log_file.flush()

            # ### Find closest topic for each doc in each community ###
            for community_num, docs in enumerate(community_docs):
                closest_topics = [lda.closest_topic(x) for x in docs]
                counter = Counter(closest_topics)
                print(sorted(counter.items()), file=log_file)
                log_file.flush()

        ### Calculate NPMI similarity across schemes ###
        npmi_scores = {}
        for config1, lda1 in lda_models.items():
            for config2, lda2 in lda_models.items():
                if config1 == config2:
                    continue

                pair_key = "{} - {}".format(config1, config2)

                topic_models1 = lda1.word_weight_tuples
                topic_models2 = lda2.word_weight_tuples

                npmi_matrix = generate_community_pairwise_similarity_matrix(topic_models1, topic_models2)
                npmi_matrix = numpy.array(npmi_matrix)

                print(npmi_matrix, file=log_file)

                avg_npmi = npmi_matrix.flatten().mean()
                npmi_scores[pair_key] = avg_npmi

                print("{} - {}".format(pair_key, avg_npmi), file=log_file)
                log_file.flush()
