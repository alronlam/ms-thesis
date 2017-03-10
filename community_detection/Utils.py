from community_detection.graph_construction import TweetGraphs
from community_detection.graph_construction import MentionGraphs
from community_detection.weight_modification.EdgeWeightModifier import *
from sentiment_analysis.evaluation import TSVParser
from twitter_data.SentiTweets import SentiTweetAdapter
from twitter_data.database import DBManager
from twitter_data.database import DBUtils
from twitter_data.parsing.csv_parser import CSVParser
from twitter_data.parsing.folders import FolderIO
from twitter_data.parsing.json_parser import JSONParser


#########################################
### Base Graph Construction Functions ###
#########################################

##############################
### User Network Functions ###
##############################

def generate_network(file_name, tweet_objects, generation_func, verbose=False):
    GRAPH_PICKLE_FILE_NAME = file_name+".pickle"
    if verbose:
        print("Going to construct the graph")
    # construct graph based on user objects
    G = generation_func(None, tweet_objects, pickle_file_name=GRAPH_PICKLE_FILE_NAME, start_index=0, verbose=verbose)
    G.save(GRAPH_PICKLE_FILE_NAME)
    return G

def generate_user_network(file_name, tweet_objects, verbose=False):
    return generate_network(file_name, tweet_objects,TweetGraphs.construct_user_graph, verbose )

def generate_user_mention_network(file_name, tweet_objects, verbose=False):
    return generate_network(file_name, tweet_objects,TweetGraphs.construct_user_mention_graph, verbose )

def generate_user_mention_hashtag_sa_network(file_name, tweet_objects, classifier, hashtag_preprocessors=[], sa_preprocessors=[], verbose=False):
    GRAPH_PICKLE_FILE_NAME = file_name+".pickle"
    if verbose:
        print("Going to construct the graph")
    # construct graph based on user objects
    G = MentionGraphs.construct_user_mention_hashtag_sa_graph(None, tweet_objects, classifier, GRAPH_PICKLE_FILE_NAME, hashtag_preprocessors=hashtag_preprocessors, sa_preprocessors=sa_preprocessors, verbose=verbose)
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
    return G

################################################
### Community Detection & Analysis Functions ###
################################################
def determine_communities(G, out_file, verbose=False):
    # Community Detection
    if verbose:
        print("Going to determine communities")
    membership = G.community_infomap(edge_weights=G.es["weight"]).membership

    # Print metrics
    modularity = G.modularity(membership)
    print("Modularity: {}".format(modularity), file=out_file)

    return membership

def remove_communities_with_less_than_n(membership, n):
    return [m for m in membership if membership.count(m) > n ]

def construct_graph_with_filtered_communities(g, membership, min_vertices_per_community):
    g = g.copy()
    valid_membership = remove_communities_with_less_than_n(membership, min_vertices_per_community)
    to_delete_ids = [v.index for v in g.vs if membership[v.index] not in valid_membership]
    g.delete_vertices(to_delete_ids)

    return (g, valid_membership)

def get_communities(membership):
    return sorted(list(set(membership)))

def get_vertex_ids_in_community(graph, membership, community_num):
    return [ index for index, x in enumerate(membership) if x == community_num]

def get_vertex_ids_in_each_community(graph, membership):
    communities = get_communities(membership)
    community_vertices = []
    for index, community in enumerate(communities):
        community_vertices.append(get_vertex_ids_in_community(graph, membership, index))
    return community_vertices

def get_user_ids_from_vertex_ids(graph, vertex_ids):
    return [vertex["name"] for vertex in graph.vs if vertex.index in vertex_ids]

def filter_tweets_belonging_to_user_ids(tweet_objects, user_ids_str):
    return [tweet.text for tweet in tweet_objects if tweet.user.id_str in user_ids_str]


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


def load_tweet_objects_from_senti_csv_files(csv_folder_path):

    USER_CSV_COL_INDEX = 1
    TEXT_CSV_COL_INDEX = 2

    csv_files = FolderIO.get_files(csv_folder_path, False, '.csv')
    csv_rows = CSVParser.parse_files_into_csv_row_generator(csv_files, True)
    senti_tweet_objects = [SentiTweetAdapter(csv_row[TEXT_CSV_COL_INDEX], csv_row[USER_CSV_COL_INDEX]) for csv_row in csv_rows]
    return senti_tweet_objects


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

def count_mentions(tweet_objects):
    count = 0
    for tweet_object in tweet_objects:
        count += len(tweet_object.entities.get('user_mentions'))
    return count

