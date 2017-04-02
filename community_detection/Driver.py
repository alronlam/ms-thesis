
import os
import pickle
from datetime import datetime

from analysis.mutual_following.FPUPC import count_mutual_edges
from analysis.topic_modelling import TopicModellerFacade
from analysis.topic_modelling.LDATopicModeller import LDATopicModeller
from analysis.word_cloud import WordCloudDriver
from analysis.word_cloud.WordCloudDriver import generate_word_cloud_per_community, get_texts_per_community
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
from sentiment_analysis.SentimentClassifier import ANEWLexiconClassifier, ContextualANEWLexiconClassifier
from sentiment_analysis.preprocessing import PreProcessing
from sentiment_analysis.preprocessing.PreProcessing import SplitWordByWhitespace, ReplaceURL, ConcatWordArray, \
    RemovePunctuationFromWords, ReplaceUsernameMention, RemoveRT, RemoveLetterRepetitions, RemoveTerm, RemoveExactTerms, \
    WordLengthFilter
from sentiment_analysis.preprocessing.PreProcessing import WordToLowercase
from twitter_data.database import DBUtils


##################### MAIN DRIVER CODE ###################

#################
### Constants ###
#################
# keras_tokenizer_pickle_path = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/feature_extraction/word_embeddings/tokenizer-vanzo_word_sequence_concat_glove_200d_preprocessed.npz.pickle"
# keras_classifier_json_path = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/neural_nets/keras_model_with_context.json"
# keras_classifier_weights_path = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/neural_nets/keras_model_with_context_weights.h5"
# keras_classifier_with_context = SentimentClassifier.KerasClassifier(keras_tokenizer_pickle_path, keras_classifier_json_path, keras_classifier_weights_path, with_context=True)
#
# keras_classifier_json_path_no_context = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/neural_nets/keras_model_no_context.json"
# keras_classifier_weights_path_no_context = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/neural_nets/keras_model_no_context_weights.h5"
# keras_classifier_no_context = SentimentClassifier.KerasClassifier(keras_tokenizer_pickle_path, keras_classifier_json_path_no_context, keras_classifier_weights_path_no_context, with_context=False)
# user_keras_sa_weight_modifier = UserVerticesSAWeightModifier(keras_classifier_with_context)
user_anew_contextual_sa_weight_modifier = UserVerticesSAWeightModifier(ContextualANEWLexiconClassifier())
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
def run_one_cycle(run_name, graph, tweet_objects, edge_weight_modifiers, topic_modelling_preprocessors=[], min_membership=100):

    run_name = "{}-{}".format(min_membership, run_name)

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

    graph.save(dir_name +"/" + run_name + "_modified.pickle")

    # Community Detection
    membership = Utils.determine_communities(graph, general_out_file, verbose=True)
    pickle.dump(membership, open(dir_name +"/" + run_name + ".membership", "wb"))
    general_out_file.flush()

    # Filter
    (graph, filtered_membership) = Utils.construct_graph_with_filtered_communities(graph, membership, min_membership)
    print("Filtered communities: {}/{}. Graph now has {} vertices and {} edges".format(len(filtered_membership), len(membership), len(graph.vs), len(graph.es)), file=general_out_file)
    membership = filtered_membership
    graph.save(dir_name +"/" + str(min_membership) + "_modified_filtered.pickle")
    pickle.dump(membership, open(dir_name +"/" + str(min_membership) + "_filtered.membership", "wb"))
    general_out_file.flush()

    # Raw texts
    Utils.generate_text_for_communities(graph, membership, tweet_objects, run_name, [])

    # print("Generating tf-idf word clouds")
    # # tf-idf
    # WordCloudDriver.generate_tfidf_word_cloud_per_community(graph,
    #                           membership,
    #                           tweet_objects,
    #                           run_name,
    #                           topic_modelling_preprocessors)


    # Topic Modelling
    print("Modelling topics")
    LDA_topic_modeller = LDATopicModeller()
    topic_modelling_out_file = open("{}/{}-topic-models.txt".format(dir_name, str(min_membership)), "w", encoding="utf-8")

    community_topics_tuple_list = TopicModellerFacade.construct_topic_models_for_communities(LDA_topic_modeller, graph, membership, tweet_objects, preprocessors=topic_modelling_preprocessors)
    for community, topics in community_topics_tuple_list:
        if topics is not None:
            print("Community {}:\n{}\n".format(community, topics), file=topic_modelling_out_file)
    topic_modelling_out_file.close()

    # Plot
    print("Plotting")
    CommunityViz.plot_communities(graph, "display_str", membership, dir_name+"/"+str(min_membership), verbose=True)

    general_out_file.close()


def run_threshold_cycle(threshold, min_membership, graph_to_load, tweet_objects, analysis_preprocessors=[]):
    try:

        base_graph_name = "threshold-{}-{}.pickle".format(threshold, graph_to_load)
        run_name = "{}-{}".format(min_membership, base_graph_name)
        general_out_file = open("{}-general-info.txt".format(run_name), "w")


        # load graph and membership

        try:
            print("Loading filtered communities")
            graph = pickle.load(open(run_name, "rb"))
            membership = pickle.load(open(run_name+".membership", "rb"))
        except Exception as e:
            print("Constructing filtered graph")
            # no filtered communities yet, try loading unfiltered membership
            graph = pickle.load(open(base_graph_name, "rb"))

            try:
                membership = pickle.load(open(base_graph_name+".membership", "rb"))
                modularity = graph.modularity(membership)
                print("Modularity: {}\n".format(modularity), file=general_out_file)
            except Exception as e:
                print("Determining membership")
                membership = Utils.determine_communities(graph, general_out_file, verbose=True)
                pickle.dump(membership, open(base_graph_name+".membership", "wb"))

            (graph, filtered_membership) = Utils.construct_graph_with_filtered_communities(graph, membership, min_membership)
            print("Filtered communities: {}/{}. Graph now has {} vertices and {} edges".format(len(filtered_membership), len(membership), len(graph.vs), len(graph.es)), file=general_out_file)
            membership = filtered_membership
            pickle.dump(graph, open(run_name, "wb"))
            pickle.dump(membership, open("{}.membership".format(run_name), "wb"))

        print("Generating raw texts")
        # Raw texts
        Utils.generate_text_for_communities(graph, membership, tweet_objects, run_name, [])
        general_out_file.close()

        print("Generating tf-idf word clouds")
        # tf-idf
        WordCloudDriver.generate_tfidf_word_cloud_per_community(graph,
                                  membership,
                                  tweet_objects,
                                  run_name,
                                  analysis_preprocessors)

        # plot
        print("Plotting")
        CommunityViz.plot_communities(graph, "display_str", membership, run_name+".png", verbose=True)

        # topic modelling
        print("Modelling topics")
        LDA_topic_modeller = LDATopicModeller()
        topic_models_file = open("{}-topic-models.txt".format(run_name), "w", encoding="utf-8")
        community_topics_tuple_list = TopicModellerFacade.construct_topic_models_for_communities(LDA_topic_modeller, graph, membership, tweet_objects, preprocessors=analysis_preprocessors)
        for community, topics in community_topics_tuple_list:
            if topics is not None:
                print("Community {}:\n{}\n".format(community, topics), file=topic_models_file)
        topic_models_file.close()

    except Exception as e:
        print(e)


brexit_topic_modelling_preprocessors = [SplitWordByWhitespace(),
                 WordToLowercase(),
                 ReplaceURL(),
                 RemovePunctuationFromWords(),
                 ReplaceUsernameMention(),
                 RemoveRT(),
                 RemoveLetterRepetitions(),
                 #WordLengthFilter(3),
                 RemoveTerm("#brexit"),
                 RemoveTerm("<url>"),
                 RemoveTerm("<username>"),
                 RemoveExactTerms(Utils.load_function_words("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/preprocessing/eng-function-words.txt")),
                 RemoveExactTerms(Utils.load_function_words("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/preprocessing/fil-function-words.txt")),
                 ConcatWordArray()]

brexit_hashtag_preprocessors = [SplitWordByWhitespace(),
                                WordToLowercase(),
                                RemoveTerm("#brexit"),
                                ConcatWordArray()]

brexit_sa_preprocessors = [] # not needed anymore as pre-processing is done inside the KerasClassifier

# json_tweet_objects = Utils.load_tweet_objects_from_json_files("D:/DLSU/Masters/MS Thesis/data-2016/test")
# json_tweet_ids = Utils.load_tweet_ids_from_json_files("D:/DLSU/Masters/MS Thesis/data-2016/test")
# json_tweet_objects = DBUtils.retrieve_all_tweet_objects_from_db(json_tweet_ids, verbose=True)

# graph = pickle.load(open("100-threshold-0.05-brexit_mention_hashtag_contextualsa_graph.pickle", "rb"))
# membership = pickle.load(open("100-threshold-0.05-brexit_mention_hashtag_contextualsa_graph.pickle.membership", "rb"))
# print("Generating raw texts")
# # Raw texts
# Utils.generate_text_for_communities(graph, membership, json_tweet_objects, "100-threshold-0.05-brexit_mention_hashtag_contextualsa_graph.pickle", [])


# run_threshold_cycle(0.05, 100, "brexit_mention_hashtag_sa_graph", json_tweet_objects, analysis_preprocessors=brexit_topic_modelling_preprocessors)

# base_graph_name = "brexit_no_rt_mention_hashtag_contextualsa_graph"
# graph = Utils.generate_user_mention_hashtag_sa_network(base_graph_name, json_tweet_objects, keras_classifier_with_context, hashtag_preprocessors=brexit_hashtag_preprocessors, sa_preprocessors=brexit_sa_preprocessors, verbose=True, load_mode=False, THRESHOLD = 0.05)
# base_graph_name = "brexit_mention_hashtag_sa_graph"
# graph = Utils.generate_user_mention_hashtag_sa_network(base_graph_name, json_tweet_objects, keras_classifier_no_context, hashtag_preprocessors=brexit_hashtag_preprocessors, sa_preprocessors=brexit_sa_preprocessors, verbose=True, load_mode=False, THRESHOLD = 0.04)
# # graph = Utils.generate_user_mention_hashtag_sa_network(base_graph_name, json_tweet_objects, keras_classifier_no_context, hashtag_preprocessors=brexit_hashtag_preprocessors, sa_preprocessors=brexit_sa_preprocessors, verbose=True, load_mode=True, THRESHOLD = 0.05)
# run_threshold_cycle(0.05, 100, base_graph_name, json_tweet_objects, analysis_preprocessors=brexit_topic_modelling_preprocessors)

# while(True):
#     threshold = float(input("Threshold?"))
#     min_membership = int(input("Min vertices in community?"))
#     graph_to_load = input("Graph to load?")
#     run_threshold_cycle(threshold, min_membership, graph_to_load, json_tweet_objects, analysis_preprocessors=brexit_topic_modelling_preprocessors)

pilipinasdebates_topic_modelling_preprocessors = [SplitWordByWhitespace(),
                 WordToLowercase(),
                 ReplaceURL(),
                 RemovePunctuationFromWords(),
                 # ReplaceUsernameMention(),
                 RemoveRT(),
                 WordLengthFilter(3),
                 RemoveLetterRepetitions(),
                 RemoveTerm("#pilipinasdebates2016"),
                 RemoveExactTerms(Utils.load_function_words("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/preprocessing/eng-function-words.txt")),
                 RemoveExactTerms(Utils.load_function_words("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/preprocessing/fil-function-words.txt")),
                 ConcatWordArray()]

pilipinasdebates_hashtag_preprocessors = [SplitWordByWhitespace(),
                                WordToLowercase(),
                                RemoveTerm("#pilipinasdebates2016"),
                                ConcatWordArray()]


pilipinasdebates_sa_preprocessors = []


#
print("Retrieving tweet objects.")
# senti_tweet_objects = Utils.load_tweet_objects_from_senti_csv_files('D:/DLSU/Masters/MS Thesis/data-2016/test', limit=100000)
pdebates_tweet_objects = Utils.load_tweet_objects_from_json_files("D:/DLSU/Masters/MS Thesis/data-2016/03/filtered")
# base_graph_name = "senti_pilipinasdebates_no_rt_mention_hashtag_contextualsa_graph"
# anew_classifier = ContextualANEWLexiconClassifier()
# graph = Utils.generate_user_mention_hashtag_sa_network(base_graph_name, senti_tweet_objects, anew_classifier, hashtag_preprocessors=pilipinasdebates_hashtag_preprocessors, sa_preprocessors=pilipinasdebates_sa_preprocessors, verbose=True, load_mode=False, THRESHOLD = 0.01)
# run_threshold_cycle(0.01, 100, base_graph_name, senti_tweet_objects, analysis_preprocessors=pilipinasdebates_hashtag_preprocessors)
# graph = Utils.generate_user_mention_hashtag_sa_network(base_graph_name, senti_tweet_objects, anew_classifier, hashtag_preprocessors=pilipinasdebates_hashtag_preprocessors, sa_preprocessors=pilipinasdebates_sa_preprocessors, verbose=True, load_mode=True, THRESHOLD = 0.007)
# run_threshold_cycle(0.007, 100, base_graph_name, senti_tweet_objects, analysis_preprocessors=pilipinasdebates_hashtag_preprocessors)

# run_threshold_cycle(0.007, 100, base_graph_name, senti_tweet_objects, analysis_preprocessors=pilipinasdebates_hashtag_preprocessors)

# while(True):
#     threshold = float(input("Threshold?"))
#     min_membership = int(input("Min vertices in community?"))
#     graph_to_load = input("Graph to load?")
#     run_threshold_cycle(threshold, min_membership, graph_to_load, senti_tweet_objects, analysis_preprocessors=pilipinasdebates_hashtag_preprocessors)


# Utils.generate_user_mention_hashtag_sa_network(base_graph_name, senti_tweet_objects, keras_classifier, hashtag_preprocessors=pilipinasdebates_hashtag_preprocessors, sa_preprocessors=pilipinasdebates_sa_preprocessors, verbose=True)
# graph = pickle.load(open(base_graph_name+".pickle", "rb"))
# run_one_cycle(base_graph_name, graph, senti_tweet_objects, [], topic_modelling_preprocessors=pilipinasdebates_topic_modelling_preprocessors) # mentions only

base_graph_name = "pdebates_mention_graph"
graph = pickle.load(open(base_graph_name+".pickle", "rb"))
# graph = Utils.generate_user_mention_network(base_graph_name, pdebates_tweet_objects, verbose=True)
run_one_cycle(base_graph_name, graph, pdebates_tweet_objects, [], topic_modelling_preprocessors=pilipinasdebates_topic_modelling_preprocessors, min_membership=300) # mentions only
run_one_cycle(base_graph_name+"_with_hashtags", graph, pdebates_tweet_objects, [user_hashtag_weight_modifier],topic_modelling_preprocessors=pilipinasdebates_topic_modelling_preprocessors, min_membership=300)
run_one_cycle(base_graph_name+"_with_hashtags_sa", graph, pdebates_tweet_objects, [user_hashtag_weight_modifier, user_anew_contextual_sa_weight_modifier], topic_modelling_preprocessors=pilipinasdebates_topic_modelling_preprocessors, min_membership=300)

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
LDA_topic_modeller = LDATopicModeller()
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
