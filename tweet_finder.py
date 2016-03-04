from pathlib import Path
from tweepy import *
import json
import tweepy
import sys
import time

def construct_reply_thread(tweet):
    reply_ancestors = list_reply_ancestors(tweet)
    # reply_successors = list_reply_successors_ids(tweet)
    return reply_ancestors + [tweet] #+ reply_successors


def list_reply_ancestors(tweet):
    ancestor_id = tweet.in_reply_to_status_id
    ancestor_tweet = None
    try:
        if ancestor_id is not None:
            ancestor_tweet = api.get_status(ancestor_id)
    except TweepError as err:
        print("Tweep Error: {}".format(err))
    except tweepy.RateLimitError:
        print("Hit the rate limit. Sleeping for 5 minutes.")
        time.sleep(60*5)
    except:
        print("Unexpected error:", sys.exc_info()[0])

    if ancestor_tweet is not None:
        return list_reply_ancestors(ancestor_tweet) + [ancestor_tweet]
    else:
        return []


def list_reply_successors_ids(tweet):
    print("stub")


def find_reply_threads(minThreadLength):
     # Initialize Some Variables
    p = Path('D:/DLSU/Masters/MS Thesis/data-2016/02/test')
    json_files = (file for file in list(p.iterdir())
                  if file.is_file())
    # Start of the Actual Program
    for json_file in json_files:
        with json_file.open() as f:
            for line in f.readlines():
                curr_tweet_json = json.loads(line)

                curr_tweet = api.get_status(curr_tweet_json["id"])

                curr_reply_thread = construct_reply_thread(curr_tweet)

                curr_reply_thread_ids = [tweet.id for tweet in curr_reply_thread]

                print("{} belongs to reply thread {} of {} length.".format(curr_tweet_json["id"], curr_reply_thread_ids, curr_reply_thread_ids.__len__()))


def find_max_retweets():
    # Initialize Some Variables
    p = Path('D:/DLSU/Masters/MS Thesis/data-2016/02/test')
    json_files = (file for file in list(p.iterdir())
                  if file.is_file())
    # Start of the Actual Program
    for json_file in json_files:

        max_retweet_count = -1
        max_retweet_id = None

        with json_file.open() as f:

            json_file_lines = f.readlines()
            print("Opening: {} - with {} tweets".format(json_file.name, json_file_lines.__len__()))

            for i in range (0, len(json_file_lines), 100):
                current_lines_group = json_file_lines[i:i+100]
                current_tweet_ids = [json.loads(line)["id"] for line in current_lines_group]

                try:
                    tweets = api.statuses_lookup(current_tweet_ids)

                    print("Successfully looked up tweets! Processing them now....")

                    for tweet in tweets:
                        if(tweet.retweet_count > max_retweet_count):
                            max_retweet_count = tweet.retweet_count
                            max_retweet_id = tweet.id
                            print("New Max {} with {} RTs".format(max_retweet_id, max_retweet_count))

                except tweepy.RateLimitError:
                    print("Hit the rate limit. Sleeping for 5 minutes.")
                    print("Current max is {} with {} RTs".format(max_retweet_id, max_retweet_count))
                    time.sleep(60*5)
                    continue
                except:
                    print("Unexpected error:", sys.exc_info()[0])

            print("Final max is {} with {} RTs".format(max_retweet_id, max_retweet_count))

find_reply_threads(3)