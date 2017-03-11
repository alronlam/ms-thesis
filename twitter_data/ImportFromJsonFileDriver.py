# json generator
# per json add if does not exist
from twitter_data.database import DBManager
from twitter_data.parsing.folders import FolderIO
from twitter_data.parsing.json_parser import JSONParser

tweet_files = FolderIO.get_files("D:/DLSU/Masters/MS Thesis/data-2016/test", False, '.json')
tweet_generator = JSONParser.parse_files_into_json_generator(tweet_files)

for index, tweet_json in enumerate(tweet_generator):
    DBManager.get_or_add_tweet_db_given_json(tweet_json)
    print("Imported {} tweets".format(index))