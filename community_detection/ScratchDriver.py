import json

# from community_detection import Utils
# from twitter_data.api import TweepyHelper
# from twitter_data.database import DBManager
# from twitter_data.parsing.folders import FolderIO
# from twitter_data.parsing.json_parser import JSONParser
#
#
#
# def construct_filtered_json(json_folder_path, limit=None):
#     tweet_files = FolderIO.get_files(json_folder_path, False, '.json')
#     tweet_generator = JSONParser.parse_files_into_json_generator(tweet_files)
#
#     tweet_jsons = []
#     index = 0
#     added_tweet_ids = set()
#     for tweet_json in tweet_generator:
#         if limit:
#             if index >= limit:
#                 break
#         index += 1
#
#         print("{}".format(index))
#         try:
#             status = TweepyHelper.parse_from_json(tweet_json)
#             #check if valid status; having limit in json means it was a rate limit response
#             if "limit" not in status._json:
#                 #check if tweet_json already contains status
#                 if status.id not in added_tweet_ids:
#                     tweet_jsons.append(tweet_json)
#                     added_tweet_ids.add(status.id)
#                     DBManager.get_or_add_tweet_db_given_json(tweet_json)
#         except Exception as e:
#             print(e)
#             pass
#     return tweet_jsons
#
#
# import pickle
#
# from community_detection import Utils
#
#

import pickle

from community_detection import Utils

root_folder = "C:/Users/user/PycharmProjects/ms-thesis/analysis/topic_modelling/graphs"
configs = [
    ("brexit_mention_graph_modified_weights", 300),
    ("brexit_mention_graph_with_hashtags_modified_weights", 300),
    ("brexit_mention_graph_with_hashtags_sa_modified_weights", 300),
    ("brexit_mention_graph_with_hashtags_contextualsa_modified_weights", 300),
    ("threshold-0.05-brexit_mention_hashtag_contextualsa_graph", 100)
]

for graph_name, min_membership in configs:
    graph = pickle.load(open("{}/{}.pickle".format(root_folder, graph_name),"rb"))
    membership = pickle.load(open("{}/{}.membership".format(root_folder, graph_name), "rb"))
    print(graph_name)
    print("{} vertices {} ({}) edges - {}".format(len(graph.vs), len(graph.es), sum(graph.es["weight"]), graph_name))
    print("Modularity: {}".format(graph.modularity(membership)))
    print()

    # membership = Utils.determine_communities(graph, None, verbose=True)
    # pickle.dump(membership, open(root_folder+"/"+graph_name+".membership", "wb"))
    #
    # (graph, membership) = Utils.construct_graph_with_filtered_communities(graph, membership, min_membership)
    # pickle.dump(graph, open("{}/{}-{}.pickle".format(root_folder, min_membership, graph_name) , "wb"))
    # pickle.dump(membership, open("{}/{}-{}.membership".format(root_folder, min_membership, graph_name), "wb"))

# tweet_objects = Utils.load_tweet_objects_from_json_files("D:/DLSU/Masters/MS Thesis/data-2016/test")
# for graph_name, min_membership in configs:
#     graph = pickle.load(open("{}/{}-{}.pickle".format(root_folder, min_membership, graph_name), "rb"))
#     membership = pickle.load(open("{}/{}-{}.membership".format(root_folder, min_membership, graph_name), "rb"))
#     Utils.generate_text_for_communities(graph, membership, tweet_objects, "{}-{}".format(min_membership, graph_name), output_dir=root_folder)

