import sys
import time
import tweepy

from tweepy import *


consumer_key = 'fwbtkGf8N97yyUZyH5YzLw'
consumer_secret = 'oQA5DunUy89Co5Hr7p4O2WmdzqiGTzssn2kMphKc8g'
access_token = '461053984-aww1IbpSVcxUE2jN8VqsOkEw8IQeEMusx4IdPM9p'
access_secret = 'WGsbat8P8flqKqyAymnWnTnAGI5hZkgdaQSE8XALs7ZEp'
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

def retrieve_tweet(tweet_id):
    func = lambda tweet_id: api.get_status(tweet_id) if tweet_id is not None else None
    return tweepy_function(func, tweet_id)

def retrieve_user(user_id):
    func = lambda user_id: api.get_user(user_id=user_id) if user_id is not None else None
    return tweepy_function(func, user_id)

def retrieve_followers_ids(user_id):
    user = retrieve_user(user_id)
    followers_ids = set()
    pages = tweepy.Cursor(api.followers_ids, id=user_id).pages()
    while(True):
        try:
            for page in pages:
                followers_ids.update(page)
                print("# of followers found: {}".format(followers_ids.__len__()))
            return followers_ids
        except tweepy.RateLimitError:
            print("Hit the Twitter API rate limit. Sleeping for 3 minutes.")
            time.sleep(60*3)
            print("Finished sleeping. Resuming execution.")
        except TweepError as err:
            print("Tweep Error: {}".format(err))
            if "http" in "{}".format(err).lower():
                print("Sleeping for 5 seconds before re-trying")
                time.sleep(5)
                print("Finished sleeping. Resuming execution.")
            else:
                break
        except Exception as e:
            print("{}".format(e))
            break

def retrieve_following_ids(user_id):
    user = retrieve_user(user_id)
    followers_ids = set()
    pages = tweepy.Cursor(api.friends_ids, id=user_id).pages()
    while(True):
        try:
            for page in pages:
                followers_ids.update(page)
                print("# of following found: {}".format(followers_ids.__len__()))
            return followers_ids
        except tweepy.RateLimitError:
            print("Hit the Twitter API rate limit. Sleeping for 3 minutes.")
            time.sleep(60*3)
            print("Finished sleeping. Resuming execution.")
        except TweepError as err:
            print("Tweep Error: {}".format(err))
            if "http" in "{}".format(err).lower():
                print("Sleeping for 5 seconds before re-trying")
                time.sleep(5)
                print("Finished sleeping. Resuming execution.")
            else:
                break
        except Exception as e:
            print("{}".format(e))
            break

def show_friendship(source_id, target_id):
    func = lambda source_id, target_id: api.show_friendship(source_id=source_id, target_id=target_id)
    return tweepy_function(func, source_id, target_id)

def tweepy_function(func, *args):
    try:
        return func(*args)
    except tweepy.RateLimitError:
        print("Hit the Twitter API rate limit. Sleeping for 5 minutes.")
        time.sleep(60*5)
        print("Finished sleeping. Resuming execution.")
        tweepy_function(func, args)
    except TweepError as err:
        print("Tweep Error: {}".format(err))
    except Exception as e:
        print("Unexpected error: {}".format(e))
