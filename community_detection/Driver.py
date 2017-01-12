from datetime import datetime

from analysis.viz import CommunityViz
from community_detection.EdgeWeightModifier import *
from community_detection.graph_construction import TweetGraphs
from twitter_data.database import DBManager
from twitter_data.parsing.folders import FolderIO
from twitter_data.parsing.json_parser import JSONParser
from sentiment_analysis import SentimentClassifier


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
    return " ".join([vertex["text"] for vertex in vertex_set ])

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


# generate_tweet_network()

def generate_user_network():

    GRAPH_PICKLE_FILE_NAME = "user-graph-{}.pickle".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    # GRAPH_PICKLE_FILE_NAME = "user-graph-2016-10-04-22-27-53.pickle"
     # Load tweets
    # use dataset with all election hashtags
    print("Reading data")
    # csv_files = FolderIO.get_files('D:/DLSU/Masters/MS Thesis/data-2016/senti_election_data/csv_files/test', False, '.csv')
    # csv_rows = CSVParser.parse_files_into_csv_row_generator(csv_files, True)
    # tweet_ids = [csv_row[0] for csv_row in csv_rows]

    tweet_files = FolderIO.get_files('D:/DLSU/Masters/MS Thesis/data-2016/test/', False, '.json')
    tweet_generator = JSONParser.parse_files_into_json_generator(tweet_files)
    # tweet_ids = [tweet["id"] for tweet in tweet_generator]
    tweet_ids = []
    for tweet in tweet_generator:
        try:
            tweet_ids.append(tweet["id"])
        except Exception as e:
            pass

    # Construct base graph (directed)
    print("Going to construct the graph")
    # G = load(GRAPH_PICKLE_FILE_NAME)
    # construct graph based on user objects
    G = TweetGraphs.construct_user_graph(None, tweet_ids, pickle_file_name=GRAPH_PICKLE_FILE_NAME, start_index=0)
    G.save(GRAPH_PICKLE_FILE_NAME)

    # Modify edge weights
    # G = SAWeightModifier(SentimentClassifier.LexiconClassifier()).modify_edge_weights(G)
    # G.save(GRAPH_PICKLE_FILE_NAME)

    # Community Detection
    print("Going to determine communities")
    community = G.community_infomap().membership

    # Plot
    print("Going to plot the graph")
    CommunityViz.plot_communities(G, "username", community)

    # Word Cloudnew_edges.__len__()
    # text_dict = combine_text_for_each_community(extract_vertices_in_communities(G, community))
    # for index, text in text_dict.items():
    #     if index == 1:
    #         break
    #     print("{}\n{}".format(index, text))

generate_user_network()
# DBManager.delete_followers_ids(461053984)
# print(len(DBManager.get_or_add_followers_ids(461053984)))
# print(len(DBManager.get_or_add_followers_ids(48284511)))
# print(len(DBManager.get_or_add_following_ids(48284511)))