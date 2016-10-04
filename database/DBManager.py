from pymongo import MongoClient
from tweets import TweepyHelper
from bson.json_util import dumps
from tweepy import *
import json
import sys

client = MongoClient('localhost', 27017)
db = client['twitter_db']
tweet_collection = db['tweet_collection']
user_collection = db['user_collection']
following_collection = db['following_collection']
followers_collection = db['followers_collection']
friendship_collection = db['friendship_collection']
lexicon_so_collection = db['lexicon_so_collection']

# Lexicon-related
def get_lexicon_so(lexicon_id):
    # return lexicon_so_collection.find_one()
    return lexicon_so_collection.find_one({"id":lexicon_id})

def add_lexicon_so_entries(lexicon_entries):
    for lexicon_entry in lexicon_entries:
        lexicon_so_collection.insert(lexicon_entry)

# Friendship-related
def get_or_add_friendship(source_id, target_id):
    if source_id > target_id:
        return get_or_add_friendship(target_id, source_id)

    friendship_identifier = str(source_id)+"-"+str(target_id)

    try:
        from_db = json.loads(dumps(friendship_collection.find_one({"id":friendship_identifier})))
        if from_db:
            return from_db

        from_api = TweepyHelper.show_friendship(source_id, target_id)
        if from_api:
            new_dict = {"id": friendship_identifier, "following": from_api[0].following, "followed_by":from_api[0].followed_by}
            friendship_collection.insert_one(new_dict)

        return new_dict

    except:
        print("Unexpected error:", sys.exc_info()[0])
        return None


# Tweet-related
def get_or_add_tweet_db_given_json(tweet_json):
    return add_or_update_db_given_json(tweet_json, tweet_collection, Status.parse)

def get_or_add_tweet(tweet_id):
    return get_or_add(tweet_id, tweet_collection, TweepyHelper.retrieve_tweet, Status.parse)

def delete_tweet(tweet_id):
    tweet_collection.delete_one({"id":tweet_id})

# User-related

def get_or_add_user(user_id):
    return get_or_add(user_id, user_collection, TweepyHelper.retrieve_user, User.parse)

def delete_user(user_id):
    user_collection.delete_one({"id":user_id})

def get_or_add_following_ids(user_id):
    return get_or_add_list(user_id, following_collection, TweepyHelper.retrieve_following_ids, 'following_ids')

def get_or_add_followers_ids(user_id):
    return get_or_add_list(user_id, followers_collection, TweepyHelper.retrieve_followers_ids, 'followers_ids')

def delete_following_ids(user_id):
    following_collection.delete_one({"id":user_id})

def delete_followers_ids(user_id):
    followers_collection.delete_one({"id":user_id})


# Helper functions

def get_or_add_list(id, collection, retrieve_func, list_name):
    try:
        from_db = collection.find_one({'id':id})
        return from_db[list_name] if from_db else add_or_update_list_db(id, collection, retrieve_func, list_name)
    except Exception as e:
        print("Get or add list exception: {}".format(e))
        return None

def add_or_update_list_db(id, collection, retrieve_func, list_name):
    from_api = retrieve_func(id)
    from_api = [str(x) for x in from_api]
    if from_api:
        json = {"id":id, list_name:from_api}
        collection.update({"id":id}, json, True)
    return from_api

def get_or_add(id, collection, retrieve_func, obj_constructor):
    try:
        from_db = json.loads(dumps(collection.find_one({"id":id})))
        return obj_constructor(TweepyHelper.api, from_db) if from_db else add_or_update_db(id, collection, retrieve_func)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return None

def add_or_update_db(id, collection, retrieve_func):
    from_api = retrieve_func(id)
    if from_api:
        collection.update({"id":id}, from_api._json, True)
    return from_api

def add_or_update_db_given_json(json, collection, obj_constructor):
    collection.update({"id":json["id"]}, json, True)
    return obj_constructor(TweepyHelper.api, json)