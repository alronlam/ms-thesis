from datetime import datetime

from analysis.viz import CommunityViz
from community_detection.EdgeWeightModifier import *

from community_detection.graph_construction import TweetGraphs

from sentiment_analysis.evaluation import TSVParser
from sentiment_analysis.machine_learning.feature_extraction.EmbeddingExtractor import EmbeddingExtractor
from twitter_data.database import DBManager
from twitter_data.parsing.csv_parser import CSVParser
from twitter_data.parsing.folders import FolderIO
from twitter_data.parsing.json_parser import JSONParser
from sentiment_analysis import SentimentClassifier


#########################
### Utility Functions ###
#########################

def extract_vertices_in_communities(graph, membership):
    dict = {}
    num_communities = len(set(membership))

    for i in range(num_communities):
        dict[i] = set()

    community_info = list(zip(graph.vs(), membership))

    for vertex, community_number in community_info:
        dict[community_number].add(vertex)

    return dict


def combine_text_for_each_community(community_dict):
    text_dict = {}
    for community_number, vertex_set in community_dict.items():
        text_dict[community_number] = combine_text_in_vertex_set(vertex_set)

    return text_dict


def combine_text_in_vertex_set(vertex_set):
    return " ".join([vertex["text"] for vertex in vertex_set])


#################################
### Dataset Loading Functions ###
#################################

def load_tweet_ids_from_vanzo_dataset():
    tsv_files = FolderIO.get_files("D:/DLSU/Masters/MS Thesis/data-2016/Context-Based_Tweets/conv_train", True, '.tsv')
    tsv_files += FolderIO.get_files("D:/DLSU/Masters/MS Thesis/data-2016/Context-Based_Tweets/conv_test", True, '.tsv')

    conversations = TSVParser.parse_files_into_conversation_generator(tsv_files)
    tweet_ids = [conversation[-1]["tweet_id"] for conversation in conversations]

    return tweet_ids


def load_tweet_ids_from_json_files(json_folder_path):
    tweet_files = FolderIO.get_files(json_folder_path, False, '.json')
    tweet_generator = JSONParser.parse_files_into_json_generator(tweet_files)
    # tweet_ids = [tweet["id"] for tweet in tweet_generator]
    tweet_ids = []
    for tweet in tweet_generator:
        try:
            tweet_ids.append(tweet["id"])
        except Exception as e:
            pass
    return tweet_ids


def load_tweet_ids_from_csv_files(csv_folder_path):
    csv_files = FolderIO.get_files(csv_folder_path, False, '.csv')
    csv_rows = CSVParser.parse_files_into_csv_row_generator(csv_files, True)
    tweet_ids = [csv_row[0] for csv_row in csv_rows]
    return tweet_ids


##########################
### Generate Functions ###
##########################

def generate_tweet_network():
    # Load tweets
    # use dataset with all election hashtags
    print("Reading data")
    tweet_files = FolderIO.get_files('D:/DLSU/Masters/MS Thesis/data-2016/03/elections/', False, '.json')
    tweet_generator = JSONParser.parse_files_into_json_generator(tweet_files)
    tweets = [DBManager.get_or_add_tweet_db_given_json(tweet) for tweet in tweet_generator]

    # Construct base graph
    print("Going to construct the graph")
    # G = load("2016-03-04-tweets-pilipinasdebates.pickle")
    G = TweetGraphs.construct_tweet_graph(None, tweets, 500, 0)
    G.save("2016-03-04-tweets-pilipinasdebates.pickle")

    # Modify edge weights
    G = SAWeightModifier(SentimentClassifier.WiebeLexiconClassifier()).modify_edge_weights(G)
    G.save("2016-03-04-tweets-pilipinasdebates.pickle")

    # Community Detection
    print("Going to determine communities")
    community = G.community_leading_eigenvector(weights="weight").membership

    # Plot
    # print("Going to plot the graph")
    # _plot(G, "text", community)

    # Word Cloud
    text_dict = combine_text_for_each_community(extract_vertices_in_communities(G, community))
    for index, text in text_dict.items():
        if index == 1:
            break
        print("{}\n{}".format(index, text))


def generate_user_network(tweet_ids):
    GRAPH_PICKLE_FILE_NAME = "user-graph-{}.pickle".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    # Construct base graph (directed)
    print("Going to construct the graph")
    # G = load(GRAPH_PICKLE_FILE_NAME)
    # construct graph based on user objects
    G = TweetGraphs.construct_user_graph(None, tweet_ids, pickle_file_name=GRAPH_PICKLE_FILE_NAME, start_index=0)
    G.save(GRAPH_PICKLE_FILE_NAME)

    # Modify edge weights

    edge_weight_modifiers = [SAWeightModifier(SentimentClassifier.AFINNLexiconClassifier())]
    G = modify_edge_weights(G, edge_weight_modifiers)
    G.save(GRAPH_PICKLE_FILE_NAME)

    # Community Detection
    print("Going to determine communities")
    community = G.community_infomap().membership

    # Plot
    print("Going to plot the graph")
    CommunityViz.plot_communities(G, "username", community)



# Load tweets
vanzo_tweet_ids = load_tweet_ids_from_vanzo_dataset()
json_tweet_ids = load_tweet_ids_from_json_files("D:/DLSU/Masters/MS Thesis/data-2016/test")

generate_user_network(vanzo_tweet_ids)
# DBManager.delete_followers_ids(461053984)
# print(len(DBManager.get_or_add_followers_ids(461053984)))
# print(len(DBManager.get_or_add_followers_ids(48284511)))
# print(len(DBManager.get_or_add_following_ids(48284511)))
#
# keras_classifier_without_context_path = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/neural_nets/keras_without_context.h5"
# keras_tokenizer_pickle_path = "C:/Users/user/PycharmProjects/ms-thesis/word_embeddings/tokenizer-vanzo_word_sequence_concat_glove_200d.npz.pickle"
# keras_feature_extractor = EmbeddingExtractor(keras_tokenizer_pickle_path)
# keras_classifier = SentimentClassifier.KerasClassifier(keras_feature_extractor, keras_classifier_without_context_path)
#
# while True:
#     print(keras_classifier.classify_sentiment(input()))