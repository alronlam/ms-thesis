# import tweepy

from TweetHelper import TweetHelper
from tweepy import *
import json
import tweepy
import sys
import time
import datetime

consumer_key = 'fwbtkGf8N97yyUZyH5YzLw'
consumer_secret = 'oQA5DunUy89Co5Hr7p4O2WmdzqiGTzssn2kMphKc8g'
access_token = '461053984-aww1IbpSVcxUE2jN8VqsOkEw8IQeEMusx4IdPM9p'
access_secret = 'WGsbat8P8flqKqyAymnWnTnAGI5hZkgdaQSE8XALs7ZEp'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

from tweepy import Stream
from tweepy.streaming import StreamListener

class MyListener(StreamListener):

    def __init__(self, desired_file_name):
        file_name = desired_file_name

    def on_data(self, data):
        try:
            with open(file_name, 'a') as f:
                f.write(data.strip()+"\n")
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        print(status)
        return True

def stream_tweets(file_name):
    twitter_stream = Stream(auth, MyListener(file_name))
    twitter_stream.filter(track=['#pilipinasdebates2016', '#phvotebinay', '#phvoteduterte', '#phvotepoe', '#phvoteroxas', '#phvotesantiago' ])

    # three overlapping bounding boxes: upper part, lower part   (to exclude malaysia), and cagayan de tawi tawi
    # twitter_stream.filter(locations=[116.09,7.26,127.19,21.34, 119.35,4.57,127.03,7.73, 118.09,6.07,122.18,9.16])

collected_ids = set()

def search_tweets(file_name):
    tweet_helper = TweetHelper()
    search_results = tweet_helper.api.search("q='#PHVoteBinay OR #PHVoteDuterte OR #PHVotePoe OR #PHVoteRoxas OR #PHVoteSantiago OR #PiliPinasDebates2016'")

    file_tweets = open(file_name, 'a')

    repeated_results = 0

    for result in search_results:
        if result.id not in collected_ids:
            file_tweets.write(json.dumps(result._json)+"\n")
            collected_ids.add(result.id)
        else:
            repeated_results += 1

    print("Repeated results count: {}".format(repeated_results))

    file_tweets.close()


file_name = 'collected_data/tweets-{}.json'.format(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
while True:

    try:
        # search_tweets(file_name)
        stream_tweets(file_name)
    except tweepy.RateLimitError:
        print("Hit the rate limit. Sleeping for 5 minutes.")
        time.sleep(60*5)
    except TweepError as err:
        print("Tweep Error: {}".format(err))
    except:
        print("Unexpected error:", sys.exc_info()[0])


