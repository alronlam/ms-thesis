import json
import sys
from traceback import print_tb

from bson.json_util import dumps
from palmettopy.palmetto import Palmetto
from pymongo import MongoClient
from tweepy import *

from twitter_data.api import TweepyHelper

client = MongoClient('localhost', 27017)
db = client['twitter_db']
tweet_collection = db['tweet_collection']
user_collection = db['user_collection']
following_collection = db['following_collection']
followers_collection = db['followers_collection']
friendship_collection = db['friendship_collection']
lexicon_so_collection = db['lexicon_so_collection']
anew_lexicon_collection = db['anew_lexicon_collection']

npmi_collection = db['npmi_collection']
palmetto = Palmetto()
UNAVAILABLE_KEY = 'unavailable'

# NPMI word-similarity related
def get_or_add_npmi(word1, word2, coherence_type="npmi"):

    word1 = word1.lower()
    word2 = word2.lower()

    if sorted([word1, word2])[0] == word2:
        temp = word1
        word1 = word2
        word2 = temp

    from_db = npmi_collection.find_one({"word1": word1, "word2":word2})
    if from_db:
        score = from_db["score"]
    else:
        score = palmetto.get_coherence([word1, word2], coherence_type=coherence_type)
        npmi_collection.insert_one({"word1":word1,"word2":word2,"score":score}, True)

    return score

# Lexicon-related
def get_lexicon_so(lexicon_id):
    # return lexicon_so_collection.find_one()
    return lexicon_so_collection.find_one({"id":lexicon_id})

def add_lexicon_so_entries(lexicon_entries):
    add_batch_entries(lexicon_so_collection, lexicon_entries)

# ANEW Lexicon-related

def get_anew_lexicon(anew_lexicon_id):
    return anew_lexicon_collection.find_one({"id":anew_lexicon_id})

def add_anew_lexicon_entries(anew_lexicon_entries):
    add_batch_entries(anew_lexicon_collection, anew_lexicon_entries)

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

def add_batch_entries(collection, entries):
     for entry in entries:
        collection.insert(entry)

def get_or_add_list(id, collection, retrieve_func, list_name):
    try:
        from_db = collection.find_one({'id':id})

        if from_db:
            # this means the API cannot give the information needed (based on a past query, so don't retry the query anymore); TODO: review this, might miss out on info not retrieved due to connectivity issues
            if UNAVAILABLE_KEY in from_db:
                return None
            else:
                return from_db[list_name]
        else:
            add_or_update_list_db(id, collection, retrieve_func, list_name)
        # return from_db[list_name] if from_db else add_or_update_list_db(id, collection, retrieve_func, list_name)
    except Exception as e:
        print("Get or add list exception: {}".format(e))
        return None

def add_or_update_list_db(id, collection, retrieve_func, list_name):
    from_api = retrieve_func(id)
    if from_api:
        from_api = [str(x) for x in from_api]
        json = {"id":id, list_name:from_api}
        collection.update({"id":id}, json, True)
    else:
        collection.update({"id":id}, {"id":id, UNAVAILABLE_KEY:True}, True)
    return from_api

def get_or_add(id, collection, retrieve_func, obj_constructor):
    try:
        from_db = json.loads(dumps(collection.find_one({"id":id})))

        if from_db:
            if UNAVAILABLE_KEY in from_db:
                return None
            else:
                return obj_constructor(TweepyHelper.api, from_db)
        else:
            add_or_update_db(id, collection, retrieve_func)
        # return obj_constructor(TweepyHelper.api, from_db) if from_db else add_or_update_db(id, collection, retrieve_func)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return None

def add_or_update_db(id, collection, retrieve_func):
    from_api = retrieve_func(id)
    if from_api:
        collection.update({"id":id}, from_api._json, True)
    else:
        collection.update({"id":id}, {"id":id, UNAVAILABLE_KEY:True}, True)
    return from_api

def add_or_update_db_given_json(json, collection, obj_constructor):
    try:
        collection.update({"id":json["id"]}, json, True)
        return obj_constructor(TweepyHelper.api, json)
    except Exception as e:
        print("Exception in add_or_update_db_given_json: {}".format(e))