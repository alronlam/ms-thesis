from pymongo import MongoClient
from tweets import TweepyHelper
from bson.json_util import dumps
from tweepy import *
import json

client = MongoClient('localhost', 27017)
db = client['twitter_db']
tweet_collection = db['tweet_collection']
user_collection = db['user_collection']

# Tweet-related

def get_or_add_tweet(tweet_id):
    return get_or_add(tweet_id, tweet_collection, TweepyHelper.retrieve_tweet, Status)

def delete_tweet(tweet_id):
    tweet_collection.delete_one({"id":tweet_id})

# User-related

def get_or_add_user(user_id):
    return get_or_add(user_id, user_collection, TweepyHelper.retrieve_user, User)

def delete_user(user_id):
    user_collection.delete_one({"id":user_id})

# Helper functions

def get_or_add(id, collection, retrieve_func, obj_constructor):
    try:
        from_db = json.loads(dumps(collection.find_one({"id":id})))
        return obj_constructor(from_db) if from_db else add_to_db(id, collection, retrieve_func)
    except:
        return None

def add_to_db(id, collection, retrieve_func):
    from_api = retrieve_func(id)
    if from_api:
        collection.insert(from_api._json)
    return from_api


