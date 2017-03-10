
import os
import pickle
from datetime import datetime

from analysis.mutual_following.FPUPC import count_mutual_edges
from analysis.topic_modelling import TopicModellerFacade
from analysis.topic_modelling.LDATopicModeller import LDATopicModeller
from community_detection import Utils
from community_detection.weight_modification.EdgeWeightModifier import *
from community_detection.weight_modification.user_graph_weight_modification.UserVerticesHashtagWeightModifier import \
    UserVerticesHashtagWeightModifier
from community_detection.weight_modification.user_graph_weight_modification.UserVerticesSAWeightModifier import \
    UserVerticesSAWeightModifier
from community_detection.weight_modification.user_graph_weight_modification.UserVerticesMentionsWeightModifier import \
    UserVerticesMentionsWeightModifier
from sentiment_analysis import SentimentClassifier
from analysis.viz import CommunityViz
from sentiment_analysis.preprocessing import PreProcessing
from sentiment_analysis.preprocessing.PreProcessing import SplitWordByWhitespace, ReplaceURL, ConcatWordArray, \
    RemovePunctuationFromWords, ReplaceUsernameMention, RemoveRT, RemoveLetterRepetitions, RemoveTerm, RemoveExactTerms
from sentiment_analysis.preprocessing.PreProcessing import WordToLowercase
from twitter_data.database import DBUtils


##################### MAIN DRIVER CODE ###################

#################
### Constants ###
#################
keras_tokenizer_pickle_path = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/feature_extraction/word_embeddings/tokenizer-vanzo_word_sequence_concat_glove_200d.npz.pickle"
keras_classifier_json_path = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/neural_nets/keras_model_no_context.json"
keras_classifier_weights_path = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/neural_nets/keras_model_no_context_weights.h5"
keras_classifier = SentimentClassifier.KerasClassifier(keras_tokenizer_pickle_path, keras_classifier_json_path, keras_classifier_weights_path)
user_keras_sa_weight_modifier = UserVerticesSAWeightModifier(keras_classifier)
user_hashtag_weight_modifier = UserVerticesHashtagWeightModifier()
# user_mention_weight_modifier = UserVerticesMentionsWeightModifier()
# tweet_keras_sa_weight_modifier = TweetVerticesSAWeightModifier(keras_classifier)


###################
### Load Tweets ###
###################
# vanzo_tweet_ids = Utils.load_tweet_ids_from_vanzo_dataset()
# vanzo_tweet_objects = DBUtils.retrieve_all_tweet_objects_from_db(vanzo_tweet_ids, verbose=True)

# json_tweet_ids = Utils.load_tweet_ids_from_json_files("D:/DLSU/Masters/MS Thesis/data-2016/test")
# json_tweet_objects = DBUtils.retrieve_all_tweet_objects_from_db(json_tweet_ids, verbose=True)
#
# senti_tweet_objects = Utils.load_tweet_objects_from_senti_csv_files('D:/DLSU/Masters/MS Thesis/data-2016/test')

#############################
### Construct Base Graphs ###
#############################
# file_name = "user-graph-{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
# generate_user_network("vanzo_user_graph", vanzo_tweet_objects, verbose=True)
# generate_tweet_hashtag_network("vanzo_tweet_hashtag_graph", vanzo_tweet_objects, keras_classifier, verbose=True)
# generate_user_mention_network("vanzo_user_mention_graph", vanzo_tweet_objects, verbose=True)

# generate_user_mention_network("brexit_user_mention_graph", json_tweet_objects, verbose=True)
# generate_user_network("brexit_user_graph", json_tweet_objects, verbose=True)

# Utils.generate_user_mention_hashtag_sa_network("senti_pilipinas_debates_mention_hashtag_sa_graph", senti_tweet_objects, keras_classifier, verbose=True)


###########################################
### User Network (Mentions) Experiments ###
###########################################
def run_one_cycle(run_name, graph, tweet_objects, edge_weight_modifiers, topic_modelling_preprocessors=[], min_membership=50):
    print("Running: "+run_name)

    # Create Output Folder
    dir_name = "{}-{}".format(run_name, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    general_out_file = open(dir_name+"/"+run_name+".txt", "w")

    print("Graph has {} vertices and {} edges\n".format(len(graph.vs), len(graph.es)), file=general_out_file)

    # Edge Weight Modification
    if edge_weight_modifiers and len(edge_weight_modifiers) > 0:
        graph = Utils.modify_network_weights(graph, run_name, tweet_objects, edge_weight_modifiers, verbose=False)
        graph.save(dir_name +"/" + run_name + "_modified_weights.pickle")

    # Community Detection
    membership = Utils.determine_communities(graph, general_out_file, verbose=True)
    pickle.dump(membership, open(dir_name +"/" + run_name + ".membership", "wb"))
    general_out_file.flush()

    # Filter
    (graph, filtered_membership) = Utils.construct_graph_with_filtered_communities(graph, membership, min_membership)
    print("Filtered communities: {}/{}. Graph now has {} vertices and {} edges".format(len(filtered_membership), len(membership), len(graph.vs), len(graph.es)), file=general_out_file)
    membership = filtered_membership
    general_out_file.flush()

    # Topic Modelling
    print("Modelling topics")
    LDA_topic_modeller = LDATopicModeller()
    out_file = open("{}/{}-topic-models.txt".format(dir_name, run_name), "w", encoding="utf-8")

    community_topics_tuple_list = TopicModellerFacade.construct_topic_models_for_communities(LDA_topic_modeller, graph, membership, tweet_objects, preprocessors=topic_modelling_preprocessors)
    for community, topics in community_topics_tuple_list:
        if topics is not None:
            print("Community {}:\n{}\n".format(community, topics), file=out_file)

    # Plot
    print("Plotting")
    CommunityViz.plot_communities(graph, "display_str", membership, dir_name+"/"+run_name, verbose=True)

    general_out_file.close()


brexit_topic_modelling_preprocessors = [SplitWordByWhitespace(),
                 WordToLowercase(),
                 ReplaceURL(),
                 RemovePunctuationFromWords(),
                 ReplaceUsernameMention(),
                 RemoveRT(),
                 RemoveLetterRepetitions(),
                 RemoveTerm("brexit"),
                 RemoveExactTerms(Utils.load_function_words("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/preprocessing/eng-function-words.txt")),
                 RemoveExactTerms(Utils.load_function_words("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/preprocessing/fil-function-words.txt")),
                 ConcatWordArray()]

brexit_hashtag_preprocessors = [SplitWordByWhitespace(),
                                WordToLowercase(),
                                RemoveTerm("#brexit"),
                                ConcatWordArray()]

#TODO double check preprocessors used for SA training
#need to remove universal hashtag(s) for sa as well
brexit_sa_preprocessors = brexit_hashtag_preprocessors

json_tweet_ids = Utils.load_tweet_ids_from_json_files("D:/DLSU/Masters/MS Thesis/data-2016/test")
json_tweet_objects = DBUtils.retrieve_all_tweet_objects_from_db(json_tweet_ids, verbose=True)
base_graph_name = "brexit_mention_hashtag_sa_graph"
Utils.generate_user_mention_hashtag_sa_network(base_graph_name, json_tweet_objects, keras_classifier, hashtag_preprocessors=brexit_hashtag_preprocessors, sa_preprocessors=brexit_sa_preprocessors, verbose=True)
graph = pickle.load(open(base_graph_name+".pickle", "rb"))
run_one_cycle(base_graph_name, graph, json_tweet_objects, [], topic_modelling_preprocessors=brexit_topic_modelling_preprocessors, min_membership=500)

# pilipinasdebates_preprocessors = [SplitWordByWhitespace(),
#                  WordToLowercase(),
#                  ReplaceURL(),
#                  RemovePunctuationFromWords(),
#                  ReplaceUsernameMention(),
#                  RemoveRT(),
#                  RemoveLetterRepetitions(),
#                  RemoveTerm("pilipinasdebates2016"),
#                  ConcatWordArray()]
#
# senti_tweet_objects = Utils.load_tweet_objects_from_senti_csv_files('D:/DLSU/Masters/MS Thesis/data-2016/test')
# base_graph_name = "senti_pilipinas_debates_mention_hashtag_sa_graph"
# Utils.generate_user_mention_hashtag_sa_network(base_graph_name, senti_tweet_objects, keras_classifier, verbose=True)
# graph = pickle.load(open(base_graph_name+".pickle", "rb"))
# run_one_cycle(base_graph_name, graph, senti_tweet_objects, [], text_preprocessors=pilipinasdebates_preprocessors) # mentions only


# run_one_cycle(base_graph_name+"_with_hashtags", graph, senti_tweet_objects, [user_hashtag_weight_modifier])
# run_one_cycle(base_graph_name+"_with_hashtags_sa", graph, senti_tweet_objects, [user_hashtag_weight_modifier, user_keras_sa_weight_modifier])
#
#
# json_tweet_ids = Utils.load_tweet_ids_from_json_files("D:/DLSU/Masters/MS Thesis/data-2016/test")
# json_tweet_objects = DBUtils.retrieve_all_tweet_objects_from_db(json_tweet_ids, verbose=True)
#
# base_graph_name = "brexit_mention_graph"
# graph = pickle.load(open(base_graph_name+".pickle", "rb"))
# run_one_cycle(base_graph_name, graph, json_tweet_objects, []) # mentions only
# run_one_cycle(base_graph_name+"_with_hashtags", graph, json_tweet_objects, [user_hashtag_weight_modifier])
# run_one_cycle(base_graph_name+"_with_hashtags_sa", graph, json_tweet_objects, [user_hashtag_weight_modifier, user_keras_sa_weight_modifier])

# vanzo_tweet_ids = Utils.load_tweet_ids_from_vanzo_dataset()
# vanzo_tweet_objects = DBUtils.retrieve_all_tweet_objects_from_db(vanzo_tweet_ids, verbose=True)
# base_graph_name = "vanzo_mention_graph"
# graph = pickle.load(open(base_graph_name+".pickle", "rb"))
# run_one_cycle(base_graph_name, graph, vanzo_tweet_objects, []) # mentions only
# run_one_cycle(base_graph_name+"_with_hashtags", graph, vanzo_tweet_objects, [user_hashtag_weight_modifier])
# run_one_cycle(base_graph_name+"_with_hashtags_sa", graph, vanzo_tweet_objects, [user_hashtag_weight_modifier, user_keras_sa_weight_modifier])



##########################################
### User Network (Follows) Experiments ###
##########################################
# TODO place output in dir for better organization

# FOLLOWS ONLY
# experiment_run_file_name = "user-graph-follows-{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
# graph = pickle.load(open("vanzo_user_graph.pickle", "rb"))
# graph = modify_network_weights(graph, experiment_run_file_name, vanzo_tweet_objects, [], verbose=True)
# graph.save("{}-modified-weights.pickle".format(experiment_run_file_name))
# membership = determine_communities(graph, experiment_run_file_name, verbose=True)
# CommunityViz.plot_communities(graph, "display_str", membership, experiment_run_file_name, verbose=True)

# FOLLOWS + MENTIONS
# experiment_run_file_name = "user-graph-follows_mentions-{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
# graph = pickle.load(open("vanzo_user_graph.pickle", "rb"))
# graph = modify_network_weights(graph, experiment_run_file_name, vanzo_tweet_objects, [user_mention_weight_modifier], verbose=False)
# graph.save("{}-modified-weights.pickle".format(experiment_run_file_name))
# membership = determine_communities(graph, experiment_run_file_name, verbose=True)
# CommunityViz.plot_communities(graph, "display_str", membership, experiment_run_file_name, verbose=True)

# FOLLOWS + MENTIONS + HASHTAGS
# experiment_run_file_name = "user-graph-follows_mentions_hashtags-{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
# graph = pickle.load(open("vanzo_user_graph.pickle", "rb"))
# graph = modify_network_weights(graph, experiment_run_file_name, vanzo_tweet_objects, [user_mention_weight_modifier, user_hashtag_weight_modifier], verbose=False)
# graph.save("{}-modified-weights.pickle".format(experiment_run_file_name))
# membership = determine_communities(graph, experiment_run_file_name, verbose=True)
# CommunityViz.plot_communities(graph, "display_str", membership, experiment_run_file_name, verbose=True)

# FOLLOWS + MENTIONS + HASHTAGS
# experiment_run_file_name = "user-graph-follows_mentions_hashtags_sa-{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
# graph = pickle.load(open("vanzo_user_graph.pickle", "rb"))
# graph = modify_network_weights(graph, experiment_run_file_name, vanzo_tweet_objects, [user_mention_weight_modifier, user_hashtag_weight_modifier, user_keras_sa_weight_modifier], verbose=False)
# graph.save("{}-modified-weights.pickle".format(experiment_run_file_name))
# membership = determine_communities(graph, experiment_run_file_name, verbose=True)
# CommunityViz.plot_communities(graph, "display_str", membership, experiment_run_file_name, verbose=True)



###################################
### Topic Modelling Experiments ###
###################################
# LDA_topic_modeller = LDATopicModeller()

def load_and_construct_topic_models(graph_pickle_file, out_file, tweet_objects, min_vertices_per_community=20):
    graph = pickle.load(open(graph_pickle_file, "rb"))
    membership = Utils.determine_communities(graph, None, verbose=True)
    (graph, membership) = Utils.construct_graph_with_filtered_communities(graph, membership, min_vertices_per_community)
    community_topics_tuple_list = TopicModellerFacade.construct_topic_models_for_communities(LDA_topic_modeller, graph, membership, tweet_objects)

    for community, topics in community_topics_tuple_list:
        if topics is not None:
            print("Community {}:\n{}\n".format(community, topics), file=out_file)



# print("PILIPINAS DEBATES")
# experiment_run_file_name = "senti_pilipinas_debates_mention_graph"
# graph = pickle.load(open("{}.pickle".format(experiment_run_file_name), "rb"))
# membership = pickle.load(open("{}_500.membership".format(experiment_run_file_name), "rb"))
#
# out_file = open("{}-topic-models.txt".format(experiment_run_file_name), "w")
# community_topics_tuple_list = TopicModellerFacade.construct_topic_models_for_communities(LDA_topic_modeller, graph, membership, senti_tweet_objects)
#
# for community, topics in community_topics_tuple_list:
#     if topics is not None:
#         print("Community {}:\n{}\n".format(community, topics), file=out_file)

# print("BREXIT")
# experiment_run_file_name = "brexit_mention_graph"
# graph = pickle.load(open("{}.pickle".format(experiment_run_file_name), "rb"))
# membership = pickle.load(open("{}_500.membership".format(experiment_run_file_name), "rb"))
#
# out_file = open("{}-topic-models.txt".format(experiment_run_file_name), "w", encoding="utf-8")
# community_topics_tuple_list = TopicModellerFacade.construct_topic_models_for_communities(LDA_topic_modeller, graph, membership, json_tweet_objects)
#
# for community, topics in community_topics_tuple_list:
#     if topics is not None:
#         print("Community {}:\n{}\n".format(community, topics), file=out_file)


# follows
# run_file_name="user-graph-follows-2017-01-24-02-46-07-modified-weights"
# out_file = open(run_file_name+"-topic-models.txt", "w")
# load_and_construct_topic_models("{}.pickle".format(run_file_name), out_file)
#
# # follows + mentions
# run_file_name="user-graph-follows_mentions-2017-01-24-02-49-47-modified-weights"
# out_file = open(run_file_name+"-topic-models.txt", "w")
# load_and_construct_topic_models("{}.pickle".format(run_file_name), out_file)
#
# # follows + mentions + hashtags
# run_file_name="user-graph-follows_mentions_hashtags-2017-01-24-02-53-37-modified-weights"
# out_file = open(run_file_name+"-topic-models.txt", "w")
# load_and_construct_topic_models("{}.pickle".format(run_file_name), out_file)
#
# # follows + mentions + hashtags + sa
# run_file_name="user-graph-follows_mentions_hashtags_sa-2017-01-24-02-56-40-modified-weights"
# out_file = open(run_file_name+"-topic-models.txt", "w")
# load_and_construct_topic_models("{}.pickle".format(run_file_name), out_file)


######### FPUPC ###########

def load_and_count_fpupc(graph_pickle_file, out_file, min_vertices_per_community=20):
    graph = pickle.load(open(graph_pickle_file, "rb"))
    membership = Utils.determine_communities(graph, None, verbose=True)
    (graph, membership) = Utils.construct_graph_with_filtered_communities(graph, membership, min_vertices_per_community)
    print("FPUPC:{}".format(count_mutual_edges(graph)), file=out_file)

# follows
# run_file_name="user-graph-follows-2017-01-24-02-46-07-modified-weights"
# out_file = open(run_file_name+"-fpupc.txt", "w")
# load_and_count_fpupc("{}.pickle".format(run_file_name), out_file)
#
# # follows + mentions
# run_file_name="user-graph-follows_mentions-2017-01-24-02-49-47-modified-weights"
# out_file = open(run_file_name+"-fpupc.txt", "w")
# load_and_count_fpupc("{}.pickle".format(run_file_name), out_file)
#
# # follows + mentions + hashtags
# run_file_name="user-graph-follows_mentions_hashtags-2017-01-24-02-53-37-modified-weights"
# out_file = open(run_file_name+"-fpupc.txt", "w")
# load_and_count_fpupc("{}.pickle".format(run_file_name), out_file)
#
# # follows + mentions + hashtags + sa
# run_file_name="user-graph-follows_mentions_hashtags_sa-2017-01-24-02-56-40-modified-weights"
# out_file = open(run_file_name+"-fpupc.txt", "w")
# load_and_count_fpupc("{}.pickle".format(run_file_name), out_file)


#################################
### Tweet Network Experiments ###
#################################
# TODO place output in dir for better organization
# experiment_run_file_name = "user-graph-{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
# graph = pickle.load(open("vanzo_user_graph.pickle", "rb"))
# graph = modify_network_weights(graph, experiment_run_file_name, vanzo_tweet_objects, [], verbose=True)
# membership = determine_communities(graph, experiment_run_file_name, verbose=True)
# CommunityViz.plot_communities(graph, "display_str", membership, experiment_run_file_name, verbose=True)
