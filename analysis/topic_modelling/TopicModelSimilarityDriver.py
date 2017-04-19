import csv
import os

import pickle

from functools import reduce

from analysis.topic_modelling.LDATopicModeller import LDATopicModeller
from community_detection import Utils
from sentiment_analysis.preprocessing import PreProcessing
from sentiment_analysis.preprocessing.PreProcessing import *
from twitter_data.database import DBManager
from twitter_data.database.DBManager import get_or_add_coherence_score
from twitter_data.parsing.folders import FolderIO

COHERENCE_TYPE = "npmi"

def load_community_docs(dir):
    csv_files = FolderIO.get_files(dir, False, '.csv')
    community_docs = []
    for csv_file in csv_files:
        with csv_file.open(encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter="\n")
            community_docs.append([row[0] for row in csv_reader if len(row) > 0])

    return community_docs
    # txt_files = FolderIO.get_files(dir, False, '.txt')
    # community_docs = []
    # for txt_file in txt_files:
    #     with txt_file.open(encoding="utf-8") as f:
    #         community_docs.append([line.strip() for line in f.readlines() if line.strip()])
    # return community_docs

def preprocess_docs(docs):
    preprocessors = [SplitWordByWhitespace(),
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
                 WordLengthFilter(3),
                 RemoveExactTerms(["amp"]),
                 RemoveExactTerms(Utils.load_function_words("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/preprocessing/eng-function-words.txt")),
                 RemoveExactTerms(Utils.load_function_words("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/preprocessing/fil-function-words.txt")),
                 ConcatWordArray()]
    return PreProcessing.preprocess_strings(docs, preprocessors)


def generate_topic_model(docs):

    if len(docs) == 0:
        return []

    lda = LDATopicModeller()
    return lda.generate_topic_models(docs)

##### TOPIC SIMILARITY #####
# def calculate_similarity_score(words):
#     while(True):
#         try:
#             url = "http://palmetto.aksw.org/palmetto-webapp/service/npmi"
#             params = {"words":" ".join(words)}
#             return float(requests.get(url, params).text)
#         except Exception as e:
#             print("Calculate similarity score exception: {}".format(e))

def generate_avg_topic_pairwise_similarity_score(topic_model1, topic_model2):

    if len(topic_model1) == 0 or len(topic_model2) == 0:
        return 0

    topic_model1 = normalize_topic_weights(topic_model1)
    topic_model2 = normalize_topic_weights(topic_model2)

    scores_weights = []

    for word_topic1, weight_topic1 in topic_model1:
        for word_topic2, weight_topic2 in topic_model2:
            similarity_score = get_or_add_coherence_score(word_topic1, word_topic2, coherence_type=COHERENCE_TYPE) #* weight_topic1 * weight_topic2
            # print("{} , {} = {}".format(word_topic1, word_topic2, similarity_score))
            scores_weights.append((similarity_score, weight_topic1 * weight_topic2) )

    total_score_weights = sum([weight for score, weight in scores_weights])
    final_score = sum([score * weight/total_score_weights for score, weight in scores_weights])

    return final_score

def generate_count_topic_pairwise_similarity_overlap(topic_model1, topic_model2):
    if len(topic_model1) == 0 or len(topic_model2) == 0:
        return 0

    total = 0

    words_topic1 = [word for word, weight in topic_model1]
    words_topic2 = [word for word, weight in topic_model2]

    word_set1 = set(words_topic1)
    word_set2 = set(words_topic2)

    intersection = word_set1.intersection(word_set2)
    union = word_set1.union(word_set2)

    # return len(intersection) / len(union)
    return len(intersection)

def normalize_topic_weights(topic_models):
    if topic_models:
        total_weights = sum([weight for word, weight in topic_models])
        return [(word, weight/total_weights) for word, weight in topic_models ]

def generate_community_pairwise_similarity_matrix(community_models1, community_models2):

    if len(community_models1) == 0 or len(community_models2) == 0:
        return []

    matrix = [[0 for x in range(len(community_models1))] for x in range(len(community_models2))]

    for topic_num1, topic_model1 in community_models1:
        for topic_num2, topic_model2 in community_models2:
            print("Generating Topic {}-{}".format(topic_num1, topic_num2))
            matrix[topic_num1][topic_num2] = generate_avg_topic_pairwise_similarity_score(topic_model1, topic_model2)

    return matrix

def to_word_string(topic_model):
    return " ".join([word for word, weight in topic_model])

def construct_rows_for_csv(similarity_matrix, row_headers, col_headers):
    csv_rows = []

    csv_rows.append(col_headers)
    for index, row in enumerate(similarity_matrix):
        row.insert(0, row_headers[index])
        csv_rows.append(row)
    return csv_rows

def generate_topic_similarities(community_topic_models, output_dir):

    similarity_matrices = []

    for index1, community_models1 in enumerate(community_topic_models):

        row_headers = [to_word_string(topic_model) for num, topic_model in community_models1]

        for index2 in range(index1+1,len(community_topic_models)):
            print("Generating Community {}-{}".format(index1, index2))
            community_models2 = community_topic_models[index2]

            if community_models1 and community_models2:
                col_headers = ["Community {}-{}".format(index1, index2)]
                col_headers.extend([to_word_string(topic_model) for num, topic_model in community_models2])

                similarity_matrix = generate_community_pairwise_similarity_matrix(community_models1, community_models2)
                similarity_matrices.append((index1, index2, similarity_matrix))

                # saving/file-writing
                pickle.dump(similarity_matrix, open("{}/{}-{}_similarity_matrix.pickle".format(output_dir, index1, index2), "wb"))
                with open("{}/{}-{}-{}_similarity_matrix.csv".format(output_dir,COHERENCE_TYPE, index1, index2), "w", newline='', encoding="utf-8") as f:
                    csv_writer = csv.writer(f)
                    csv_rows = construct_rows_for_csv(similarity_matrix, row_headers, col_headers)
                    csv_writer.writerows(csv_rows)

    return similarity_matrices


def save_topic_similarities(topic_similarity_matrices, output_dir, graph_scheme):
    pickle.dump(topic_similarity_matrices, open("{}/{}_similarity_matrices.pickle".format(output_dir, graph_scheme), "wb"))
    # for community_num1, community_num2, topic_similarity_matrix in topic_similarity_matrices:
    #     with open("{}/{}-{}_similarity_matrix.csv".format(output_dir, community_num1, community_num2), "w", newline='', encoding="utf-8") as f:
    #         csv_writer = csv.writer(f)
    #         csv_writer.writerows(topic_similarity_matrix)


def generate_string_for_topics(topics):
    topic_string = []
    for topic_num, word_weight_tuples in topics:
            topic_string.append("\nTopic {}".format(topic_num))
            topic_string.extend(["{} - {}".format(word, weight) for word, weight in word_weight_tuples])
    return "\n".join(topic_string)


def save_topic_models_to_readable_format_if_not_exists(community_topic_models, file_path):
    try:
        open(file_path, "r", encoding="utf-8")
    except Exception as e:
        topic_models_file = open(file_path, "w", encoding="utf-8")
        for index, community_topic_model in enumerate(community_topic_models):
            print("Community {}:\n{}\n\n".format(index, generate_string_for_topics(community_topic_model)), file=topic_models_file)
        topic_models_file.close()

def clean_topic_models(community_topic_models):
    def remove_ellipsis(word):
        return re.sub("…", "", word)

    for index, community_topic_model in enumerate(community_topic_models):
        for topic_num, word_weight_tuples in community_topic_model:
            cleaned = [(remove_ellipsis(word), weight) for word, weight in word_weight_tuples]
            community_topic_model[topic_num] = (topic_num, cleaned)
    return community_topic_models

def count_match(similarity_matrix, threshold=0):
    count = 0
    for row in similarity_matrix:
        for col in row:
            if col > threshold:
                count += 1
    return count

def remove_headers_from_similarity_matrix(similarity_matrix):
    similarity_matrix = similarity_matrix[1:]
    similarity_matrix = [row[1:] for row in similarity_matrix]
    return similarity_matrix

def generate_summary_matrix(similarity_matrices):
    summary_matrix = [[0 for x in range(6)] for x in range(6) ]

    for index1, index2, similarity_matrix in similarity_matrices:
        similarity_matrix = remove_headers_from_similarity_matrix(similarity_matrix)
        summary_matrix[index1][index2] = count_match(similarity_matrix, 0.1)

    return summary_matrix

def common_word_string(similarity_matrix, threshold=0):
    count = 0
    common_words = set()
    for row_index in range(1, len(similarity_matrix)):
        row = similarity_matrix[row_index]
        for col_index in range(1, len(row)):
            col = row[col_index]
            if col > threshold:
                count += 1
                word_set1 = set(similarity_matrix[row_index][0].split())
                word_set2 = set(similarity_matrix[0][col_index].split())
                common_words = common_words.union(word_set1.intersection(word_set2))
                # common_words = common_words.union(word_set1)
                # common_words = common_words.union(word_set2)


    common_words_string = " ".join(list(common_words)) if len(common_words) > 0 else "None"
    return common_words_string, count

def generate_summary_common_word_matrix(similarity_matrices):
    word_matrix = [["None" for x in range(6)] for x in range(6) ]
    count_matrix = [[0 for x in range(6)] for x in range(6) ]

    for index1, index2, similarity_matrix in similarity_matrices:
        word_matrix[index1][index2], count_matrix[index1][index2] = common_word_string(similarity_matrix, 0)

    return word_matrix, count_matrix

def get_words_from_topic(word_weight_tuples):
    return " ".join([word for word, weight in word_weight_tuples])

def flatten_to_1d(similarity_matrix):
    flattened = []
    for row in similarity_matrix[1:]:
        flattened.extend(row[1:])
    return flattened

root_folder="graphs"
dirs = [
        "mentions",
        "hashtags",
        "sa",
        "contextualsa",
        # "scoring-sa",
        "scoring-contextualsa"
        ]

# #generate topic models
# for dir in dirs:
#     print("Loading files")
#     community_docs = load_community_docs(root_folder+"/"+dir)
#     print("Preprocessing")
#     community_docs = [preprocess_docs(community_doc) for community_doc in community_docs]
#
#     #check first if it exists already
#     try:
#         print("Loading topic models")
#         community_topic_models = pickle.load(open(root_folder+"/"+dir+"/"+dir+"-topic_models.pickle", "rb"))
#     except Exception as e:
#         print("Generating topic models")
#         community_topic_models = [generate_topic_model(community_doc) for community_doc in community_docs]
#         pickle.dump(community_topic_models, open(root_folder+"/"+dir+"/"+dir+"-topic_models.pickle", "wb"))
#
#     save_topic_models_to_readable_format_if_not_exists(community_topic_models,"{}/{}/{}-topic-models-cleaned.txt".format(root_folder, dir, dir) )
#     print("Generating topic similarities")
#     topic_similarities = generate_topic_similarities(community_topic_models, root_folder+"/"+dir)
#     save_topic_similarities(topic_similarities, root_folder+"/"+dir, dir)
#
#
# # Driver code to insert missing header in saved similarity matrices
# for dir in dirs:
#     print("Loading files")
#     similarity_matrices = pickle.load(open("{}/{}/{}_similarity_matrices.pickle".format(root_folder, dir, dir), "rb"))
#     print("Loading topic models")
#     community_topic_models = pickle.load(open(root_folder+"/"+dir+"/"+dir+"-topic_models.pickle", "rb"))
#
#     topic_model_words = [[""]+[get_words_from_topic(topic) for topic_num, topic in topic_model] for topic_model in community_topic_models]
#     topic_model_words = [word_list for word_list in topic_model_words if len(word_list) > 1]
#     indices = [0,1,2,3,1,2,3,2,3,3]
#     print(topic_model_words)
#     for matrix_index, (index1, index2, similarity_matrix) in enumerate(similarity_matrices):
#         similarity_matrix.insert(0, topic_model_words[indices[matrix_index]])
#         # print(topic_model_words[indices[matrix_index]])
#     pickle.dump(similarity_matrices, open("{}/{}/{}_similarity_matrices_new.pickle".format(root_folder, dir, dir), "wb"))
#
#
# # generate summary matrices
for dir in dirs:
    similarity_matrices = pickle.load(open("{}/{}/Raw Word Overlap/{}_similarity_matrices_new.pickle".format(root_folder, dir, dir), "rb"))
    # summary_matrix = generate_summary_matrix(similarity_matrices)
    # pickle.dump(summary_matrix, open("{}/{}/{}_summary_matrix.pickle".format(root_folder, dir, dir), "wb"))
    # with open("{}/{}/{}_summary_matrix.csv".format(root_folder, dir, dir), "w", encoding="utf-8", newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     csv_writer.writerows(summary_matrix)

    flattened_similarity_matrices = [flatten_to_1d(similarity_matrix) for index1, index2, similarity_matrix in similarity_matrices]
    with open("{}/{}/Raw Word Overlap/{}_flattened_similarity_matrices_word_count.csv".format(root_folder, dir, dir), "w", encoding="utf-8", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(flattened_similarity_matrices)
#
#     # word_matrix, count_matrix = generate_summary_common_word_matrix(similarity_matrices)
#     # pickle.dump(word_matrix, open("{}/{}/{}_word_matrix.pickle".format(root_folder, dir, dir), "wb"))
#     # with open("{}/{}/{}_word_matrix.csv".format(root_folder, dir, dir), "w", encoding="utf-8", newline='') as csv_file:
#     #     csv_writer = csv.writer(csv_file)
#     #     csv_writer.writerows(word_matrix)
#     #
#     # pickle.dump(count_matrix, open("{}/{}/{}_count_matrix.pickle".format(root_folder, dir, dir), "wb"))
#     # with open("{}/{}/{}_count_matrix.csv".format(root_folder, dir, dir), "w", encoding="utf-8", newline='') as csv_file:
#     #     csv_writer = csv.writer(csv_file)
#     #     csv_writer.writerows(count_matrix)



def retain_only_words_in_community_model(community_model):
    for topic_model in community_model:
        for (topic_num, word_weight_tuples) in topic_model:
            topic_model[topic_num] = [word for word, weight in word_weight_tuples]

    return community_model

def get_unique_words(word_sets):
    final_list = []
    for i, word_set in enumerate(word_sets):
        other_words = set()

        for j in range(len(word_sets)):
            if i!=j:
                other_words = other_words.union(word_sets[j])

        unique = word_set - other_words
        final_list.append(unique)
    return final_list

def join_lists_into_one(lists):
    new_list = []
    for list in lists:
        new_list.extend(list)
    return new_list

def process_community_models():
    for dir in dirs:
        community_models = pickle.load(open(root_folder+"/"+dir+"/"+dir+"-topic_models.pickle", "rb"))
        community_models = [x for x in community_models if len(x) > 0]
        community_models = retain_only_words_in_community_model(community_models)
        community_models_set = [join_lists_into_one(x) for x in community_models]
        community_models_set = [set(x) for x in community_models_set]
        unique_word_sets = get_unique_words(community_models_set)

        for community_index, community_model in enumerate(community_models):

            for index, topic_words in enumerate(community_model):
                community_model[index] = [word.upper()  if (word in unique_word_sets[community_index]) else word for word in topic_words]

        for index, topic_model in enumerate(community_models):
            with open("{}/{}_unique_words_{}.txt".format(root_folder, dir, index), "w", encoding="utf-8", newline='') as f:
                words = "\n".join([" ".join(word_list) for word_list in topic_model])
                f.write(words)

        with open("{}/{}/{}_unique_words.csv".format(root_folder, dir, dir), "w", encoding="utf-8", newline='') as f:
            csv_writer = csv.writer(f)
            community_models = [[" ".join(word_list) for word_list in topic_model] for topic_model in community_models]
            csv_writer.writerows(list(map(list, zip(*community_models))))

process_community_models()

def convert_community_model_to_word_set(community_model):
    words = set()
    for topic_model in community_model:
        for topic_num, word_weight_tuples in topic_model:
            words = words.union(set([word for word, weight in word_weight_tuples]))
    return words


def process_community_models_as_a_whole():
    schemes = []
    for dir in dirs:
        schemes.append(pickle.load(open(root_folder+"/"+dir+"/"+dir+"-topic_models.pickle", "rb")))

    scheme_word_sets = [convert_community_model_to_word_set(x) for x in schemes]
    unique_word_sets = get_unique_words(scheme_word_sets)

    for scheme_index, scheme_word_set in enumerate(scheme_word_sets):
        new_word_set = set()
        unique_count = 0
        for word in scheme_word_set:
            new_word_set.add(word.upper() if word in unique_word_sets[scheme_index] else word)
            if word in unique_word_sets[scheme_index]:
                unique_count +=  1
        print("Scheme {} - Unique {}".format(dirs[scheme_index],unique_count ))
        scheme_word_sets[scheme_index] = new_word_set

    for index, scheme in enumerate(scheme_word_sets):
        with open("{}/{}_unique_words.txt".format(root_folder, dirs[index]), "w", encoding="utf-8", newline='') as f:
            # csv_writer = csv.writer(f)
            scheme_words = " ".join(list(scheme))
            # csv_writer.writerow(scheme_words)
            f.write(scheme_words)
    with open("{}/schemes_unique_words.csv".format(root_folder), "w", encoding="utf-8", newline='') as f:
        csv_writer = csv.writer(f)
        scheme_words = [" ".join(list(x)) for x in scheme_word_sets]
        csv_writer.writerow(scheme_words)
        # csv_writer.writerow(list(map(list, zip(*scheme_words))))

# process_community_models_as_a_whole()

def input_unique_words():

    num_sets = int(input("Num sets: "))
    word_sets = [set(input("Enter word set {}: ".format(i)).split()) for i in range(num_sets)]
    print()
    final_list = get_unique_words(word_sets)
    for unique_set in final_list:
        print(" ".join(list(unique_set)))

# input_unique_words()