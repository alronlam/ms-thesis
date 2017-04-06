import csv

import pickle
from random import randint

from random import randrange
import requests

from analysis.topic_modelling.LDATopicModeller import LDATopicModeller
from community_detection import Utils
from sentiment_analysis.preprocessing import PreProcessing
from sentiment_analysis.preprocessing.PreProcessing import *
from twitter_data.database import DBManager
from twitter_data.database.DBManager import get_or_add_npmi
from twitter_data.parsing.folders import FolderIO


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
                 RemovePunctuationFromWords(),
                 ReplaceUsernameMention(),
                 RemoveRT(),
                 RemoveLetterRepetitions(),
                 WordLengthFilter(3),
                 RemoveTerm("#brexit"),
                 RemoveTerm("<url>"),
                 RemoveTerm("<username>"),
                 RemoveExactTerms(Utils.load_function_words("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/preprocessing/eng-function-words.txt")),
                 RemoveExactTerms(Utils.load_function_words("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/preprocessing/fil-function-words.txt")),
                 ConcatWordArray()]
    return PreProcessing.preprocess_strings(docs, preprocessors)


def generate_topic_model(docs):
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
    total = 0

    for word_topic1, weight_topic1 in topic_model1:
        for word_topic2, weight_topic2 in topic_model2:
            similarity_score = get_or_add_npmi(word_topic1, word_topic2) * weight_topic1 * weight_topic2
            print("{} , {} = {}".format(word_topic1, word_topic2, similarity_score))
            total += similarity_score
    return total / (len(topic_model1)* len(topic_model2))

def generate_community_pairwise_similarity_matrix(community_models1, community_models2):
    matrix = [[0 for x in range(len(community_models1))] for x in range(len(community_models2))]

    for topic_num1, topic_model1 in community_models1:
        for topic_num2, topic_model2 in community_models2:
            print("Generating Topic {}-{}".format(topic_num1, topic_num2))
            matrix[topic_num1][topic_num2] = generate_avg_topic_pairwise_similarity_score(topic_model1, topic_model2)

    return matrix

def generate_topic_similarities(community_topic_models):

    similarity_matrices = []

    for index1, community_models1 in enumerate(community_topic_models):
        for index2 in range(index1+1,len(community_topic_models)):
            print("Generating Community {}-{}".format(index1, index2))
            community_models2 = community_topic_models[index2]
            similarity_matrix = generate_community_pairwise_similarity_matrix(community_models1, community_models2)
            similarity_matrices.append((index1, index2, similarity_matrix))
    return similarity_matrices


def save_topic_similarities(topic_similarity_matrices, output_dir, graph_scheme):
    pickle.dump(topic_similarities, open("{}/{}_similarity_matrices.pickle".format(output_dir, graph_scheme), "wb"))
    for community_num1, community_num2, topic_similarity_matrix in topic_similarity_matrices:
        with open("{}/{}-{}_similarity_matrix.csv".format(output_dir, community_num1, community_num2), encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(topic_similarity_matrix)


# print(calculate_similarity_score(["apple","banana"]))


root_folder="graphs"
dirs = [
        "mentions",
        "hashtags",
        "sa",
        "contextualsa",
        "scoring"
        ]

# #generate topic models
# for dir in dirs:
#     print("Loading files")
#     community_docs = load_community_docs(root_folder+"/"+dir)
#     print("Preprocessing")
#     community_docs = [preprocess_docs(community_doc) for community_doc in community_docs]
#     print("Generating topic models")
#     community_topic_models = [generate_topic_model(community_doc) for community_doc in community_docs]
#     pickle.dump(community_topic_models, open(root_folder+"/"+dir+"/"+dir+"-topic_models.pickle", "wb"))

# generate topic similarity scores
for dir in dirs:
    community_topic_models = pickle.load(open(root_folder+"/"+dir+"/"+dir+"-topic_models.pickle", "rb"))
    # print(community_topic_models[0])
    topic_similarities = generate_topic_similarities(community_topic_models)
    save_topic_similarities(topic_similarities, root_folder+"/"+dir, dir)