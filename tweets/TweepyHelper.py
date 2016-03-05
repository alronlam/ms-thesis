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
    tweet = None
    try:
        if tweet_id is not None:
            tweet = api.get_status(tweet_id)
    except tweepy.RateLimitError:
        print("Hit the rate limit. Sleeping for 5 minutes.")
        time.sleep(60*5)
        tweet = retrieve_tweet(tweet_id)
    except TweepError as err:
        print("Tweep Error: {}".format(err))
    except:
        print("Unexpected error:", sys.exc_info()[0])

    return tweet