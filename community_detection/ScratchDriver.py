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
#
#
# # filtered_file = open("D:/DLSU/Masters/MS Thesis/data-2016/03/tweets-2016-03-20 12-56-08-pilipinasdebates_filtered.json", "w", encoding="utf8")
#
# pilipinasdebates_tweet_jsons = construct_filtered_json('D:/DLSU/Masters/MS Thesis/data-2016/03')
# print(len(pilipinasdebates_tweet_jsons))
# # filtered_file.writelines([json.dumps(tweet) +"\n" for tweet in pilipinasdebates_tweet_jsons])
# # filtered_file.close()
from community_detection import Utils
from twitter_data.database import DBUtils

tweets = Utils.load_tweet_objects_from_json_files("D:/DLSU/Masters/MS Thesis/data-2016/03")
print("Unfiltered length: {}".format(len(tweets)))

tweets = Utils.load_tweet_objects_from_json_files("D:/DLSU/Masters/MS Thesis/data-2016/03/filtered")
print("Filtered length: {}".format(len(tweets)))
