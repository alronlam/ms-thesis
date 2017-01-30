import pickle
from datetime import datetime

from analysis.topic_modelling import TopicModellerFacade
from analysis.topic_modelling.LDATopicModeller import LDATopicModeller
from analysis.viz import CommunityViz
from community_detection import Utils
from community_detection.graph_construction import TweetGraphs
from community_detection.weight_modification.EdgeWeightModifier import *
from community_detection.weight_modification.user_graph_weight_modification.UserVerticesHashtagWeightModifier import \
    UserVerticesHashtagWeightModifier
from community_detection.weight_modification.user_graph_weight_modification.UserVerticesSAWeightModifier import \
    UserVerticesSAWeightModifier
from community_detection.weight_modification.user_graph_weight_modification.UserVerticesMentionsWeightModifier import \
    UserVerticesMentionsWeightModifier
from sentiment_analysis import SentimentClassifier
from sentiment_analysis.evaluation import TSVParser
from twitter_data.database import DBManager
from twitter_data.database import DBUtils
from twitter_data.parsing.csv_parser import CSVParser
from twitter_data.parsing.folders import FolderIO
from twitter_data.parsing.json_parser import JSONParser


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
        if len(vertex_set) > 5:
            text_dict[community_number] = combine_text_in_vertex_set(vertex_set)

    return text_dict


def combine_text_in_vertex_set(vertex_set):
    return " ".join([vertex["display_str"] for vertex in vertex_set])

def collect_following_followers(tweet_ids):
     # Retrieve tweets from the DB
    tweet_objects = DBUtils.retrieve_all_tweet_objects_from_db(tweet_ids, verbose=True)
    for index, tweet_obj in enumerate(tweet_objects):
        user_id_str = tweet_obj.user.id_str
        follower_ids = DBManager.get_or_add_followers_ids(user_id_str)
        following_ids = DBManager.get_or_add_following_ids(user_id_str)

        print("Collecting following/followers: Processed {}/{} tweets.".format(index+1, len(tweet_objects)))

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


#########################################
### Base Graph Construction Functions ###
#########################################

##############################
### User Network Functions ###
##############################
def generate_user_network(file_name, tweet_objects, verbose=False):
    GRAPH_PICKLE_FILE_NAME = file_name+".pickle"
    if verbose:
        print("Going to construct the graph")
    # construct graph based on user objects
    G = TweetGraphs.construct_user_graph(None, tweet_objects, pickle_file_name=GRAPH_PICKLE_FILE_NAME, start_index=0, verbose=verbose)
    G.save(GRAPH_PICKLE_FILE_NAME)
    return G

###############################
### Tweet Network Functions ###
###############################
def generate_tweet_hashtag_network(file_name, tweet_objects, sentiment_classifier, verbose=False):
    GRAPH_PICKLE_FILE_NAME = "{}.pickle".format(file_name)

    # Construct base graph
    print("Going to construct the graph")
    G = TweetGraphs.construct_tweet_hashtag_graph_with_sentiment(None, tweet_objects, GRAPH_PICKLE_FILE_NAME, sentiment_classifier)
    G.save(GRAPH_PICKLE_FILE_NAME)
    return G

##########################################
### Edge Weight Modification Functions ###
##########################################
def modify_network_weights(G, file_name, tweet_objects, edge_weight_modifiers, verbose=False):
    # Modify edge weights
    if verbose:
        print("Going to modify edge weights")
    G = modify_edge_weights(G, edge_weight_modifiers, {"tweets":tweet_objects}, verbose)
    G.save(file_name+".pickle")
    return G

################################################
### Community Detection & Analysis Functions ###
################################################
def determine_communities(G, file_name, verbose=False):
    # Community Detection
    if verbose:
        print("Going to determine communities")
    membership = G.community_infomap(edge_weights=G.es["weight"]).membership

    # Print metrics
    modularity = G.modularity(membership)
    print("Modularity: {}".format(modularity))

    if file_name:
        out_file = open("{}".format(file_name+".txt"), "w" )
        out_file.write("Modularity: {}".format(modularity))
        out_file.close()

    return membership


##################### MAIN DRIVER CODE ###################


#################
### Load Tweets ###
#################
vanzo_tweet_ids = load_tweet_ids_from_vanzo_dataset()
vanzo_tweet_objects = DBUtils.retrieve_all_tweet_objects_from_db(vanzo_tweet_ids[:100], verbose=True)

# json_tweet_ids = load_tweet_ids_from_json_files("D:/DLSU/Masters/MS Thesis/data-2016/test")
# json_tweet_objects = DBUtils.retrieve_all_tweet_objects_from_db(json_tweet_ids, verbose=True)

#################
### Constants ###
#################
keras_tokenizer_pickle_path = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/feature_extraction/word_embeddings/tokenizer-vanzo_word_sequence_concat_glove_200d.npz.pickle"
keras_classifier_json_path = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/neural_nets/keras_model_no_context.json"
keras_classifier_weights_path = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/neural_nets/keras_model_no_context_weights.h5"
# keras_classifier = SentimentClassifier.KerasClassifier(keras_tokenizer_pickle_path, keras_classifier_json_path, keras_classifier_weights_path)
# user_keras_sa_weight_modifier = UserVerticesSAWeightModifier(keras_classifier)
# user_hashtag_weight_modifier = UserVerticesHashtagWeightModifier()
# user_mention_weight_modifier = UserVerticesMentionsWeightModifier()
# tweet_keras_sa_weight_modifier = TweetVerticesSAWeightModifier(keras_classifier)

#############################
### Construct Base Graphs ###
#############################
# file_name = "user-graph-{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
# generate_user_network("vanzo_user_graph", vanzo_tweet_objects, verbose=True)
# generate_tweet_hashtag_network("vanzo_tweet_hashtag_graph", vanzo_tweet_objects, keras_classifier, verbose=True)

################################
### User Network Experiments ###
################################
# TODO place output in dir for better organization

# FOLLOWS ONLY
# experiment_run_file_name = "user-graph-follows-{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
# graph = pickle.load(open("vanzo_user_graph.pickle", "rb"))
# graph = modify_network_weights(graph, experiment_run_file_name, vanzo_tweet_objects, [], verbose=True)
# graph.save("{}-modified-weights.pickle".format(experiment_run_file_name))
# membership = determine_communities(graph, experiment_run_file_name, verbose=True)
# CommunityViz.plot_communities(graph, "display_str", membership, experiment_run_file_name, verbose=True)
#
# # FOLLOWS + MENTIONS
# experiment_run_file_name = "user-graph-follows_mentions-{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
# graph = pickle.load(open("vanzo_user_graph.pickle", "rb"))
# graph = modify_network_weights(graph, experiment_run_file_name, vanzo_tweet_objects, [user_mention_weight_modifier], verbose=True)
# graph.save("{}-modified-weights.pickle".format(experiment_run_file_name))
# membership = determine_communities(graph, experiment_run_file_name, verbose=True)
# CommunityViz.plot_communities(graph, "display_str", membership, experiment_run_file_name, verbose=True)
#
# # FOLLOWS + MENTIONS + HASHTAGS
# experiment_run_file_name = "user-graph-follows_mentions_hashtags-{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
# graph = pickle.load(open("vanzo_user_graph.pickle", "rb"))
# graph = modify_network_weights(graph, experiment_run_file_name, vanzo_tweet_objects, [user_mention_weight_modifier, user_hashtag_weight_modifier], verbose=True)
# graph.save("{}-modified-weights.pickle".format(experiment_run_file_name))
# membership = determine_communities(graph, experiment_run_file_name, verbose=True)
# CommunityViz.plot_communities(graph, "display_str", membership, experiment_run_file_name, verbose=True)
#
# # FOLLOWS + MENTIONS + HASHTAGS
# experiment_run_file_name = "user-graph-follows_mentions_hashtags_sa-{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
# graph = pickle.load(open("vanzo_user_graph.pickle", "rb"))
# graph = modify_network_weights(graph, experiment_run_file_name, vanzo_tweet_objects, [user_mention_weight_modifier, user_hashtag_weight_modifier, user_keras_sa_weight_modifier], verbose=True)
# graph.save("{}-modified-weights.pickle".format(experiment_run_file_name))
# membership = determine_communities(graph, experiment_run_file_name, verbose=True)
# CommunityViz.plot_communities(graph, "display_str", membership, experiment_run_file_name, verbose=True)


### Construct topic models
LDA_topic_modeller = LDATopicModeller()

def load_and_construct_topic_models(graph_pickle_file, out_file, min_vertices_per_community=20):
    graph = pickle.load(open(graph_pickle_file, "rb"))
    membership = determine_communities(graph, None, verbose=True)
    (graph, membership) = Utils.construct_graph_with_filtered_communities(graph, membership, min_vertices_per_community)
    community_topics_tuple_list = TopicModellerFacade.construct_topic_models_for_communities(LDA_topic_modeller, graph, membership, vanzo_tweet_objects)

    for community, topics in community_topics_tuple_list:
        if topics is not None:
            print("Community {}: {}\n".format(community, topics), file=out_file)



# follows + mentions + hashtags + sa
run_file_name="user-graph-follows_mentions_hashtags_sa-2017-01-24-02-56-40"
out_file = open(run_file_name+"-topic-models.txt", "w")
load_and_construct_topic_models("{}.pickle".format(run_file_name), out_file)

# follows
run_file_name="user-graph-follows-2017-01-24-02-46-07-modified-weights"
out_file = open(run_file_name+"-topic-models.txt", "w")
load_and_construct_topic_models("user-graph-follows-2017-01-24-02-46-07-modified-weights.pickle", out_file)




#################################
### Tweet Network Experiments ###
#################################
# TODO place output in dir for better organization
# experiment_run_file_name = "user-graph-{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
# graph = pickle.load(open("vanzo_user_graph.pickle", "rb"))
# graph = modify_network_weights(graph, experiment_run_file_name, vanzo_tweet_objects, [], verbose=True)
# membership = determine_communities(graph, experiment_run_file_name, verbose=True)
# CommunityViz.plot_communities(graph, "display_str", membership, experiment_run_file_name, verbose=True)
