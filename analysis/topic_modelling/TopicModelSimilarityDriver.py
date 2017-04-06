import csv

import pickle

import requests

from analysis.topic_modelling.LDATopicModeller import LDATopicModeller
from community_detection import Utils
from sentiment_analysis.preprocessing import PreProcessing
from sentiment_analysis.preprocessing.PreProcessing import *
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

def calculate_similarity_score(words):
    url = "http://palmetto.aksw.org/palmetto-webapp/service/npmi"
    params = {"words":" ".join(words)}
    return float(requests.get(url, params).text)

def generate_topic_similarities(topic_models):

    pass

def save_topic_similarities(topic_similarities):
    pass

print(calculate_similarity_score(["apple","banana"]))


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
# for dir in dirs:
#     community_topic_models = pickle.load(open(root_folder+"/"+dir+"/"+dir+"-topic_models.pickle", "rb"))
#     topic_similarities = generate_topic_similarities(community_topic_models)
#     save_topic_similarities(topic_similarities)