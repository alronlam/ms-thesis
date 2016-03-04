import sys
import time

import tweepy
from tweepy import *

from tweets.Node import Node


class TweetHelper:

    def __init__(self):
        # Tweepy Variables
        consumer_key = 'fwbtkGf8N97yyUZyH5YzLw'
        consumer_secret = 'oQA5DunUy89Co5Hr7p4O2WmdzqiGTzssn2kMphKc8g'
        access_token = '461053984-aww1IbpSVcxUE2jN8VqsOkEw8IQeEMusx4IdPM9p'
        access_secret = 'WGsbat8P8flqKqyAymnWnTnAGI5hZkgdaQSE8XALs7ZEp'
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)
        self.api = tweepy.API(auth)

    def list_reply_ancestors(self, tweet):

        ancestor_id = tweet.in_reply_to_status_id
        ancestor_tweet = self.retrieve_tweet(ancestor_id)

        if ancestor_tweet is None:
            return [tweet]
        else:
            return self.list_reply_ancestors(ancestor_tweet) + [tweet]


    def find_root_ancestor(self, tweet):
        ancestor_id = tweet.in_reply_to_status_id
        ancestor_tweet = self.retrieve_tweet(ancestor_id)

        if ancestor_tweet is None:
            return tweet
        else:
            return self.find_root_ancestor(ancestor_tweet)


    def retrieve_tweet(self, tweet_id):
        tweet = None
        try:
            if tweet_id is not None:
                tweet = self.api.get_status(tweet_id)
        except tweepy.RateLimitError:
            print("Hit the rate limit. Sleeping for 5 minutes.")
            time.sleep(60*5)
            tweet = self.retrieve_tweet(tweet_id)
        except TweepError as err:
            print("Tweep Error: {}".format(err))
        except:
            print("Unexpected error:", sys.exc_info()[0])

        return tweet


    def construct_reply_tree(self, tweet):

        # Search for replies to the given tweet
        raw_search_results = self.api.search("q='to:{}'".format(tweet.user.screen_name), sinceId = tweet.id)
        filtered_search_results = [result for result in raw_search_results if result.in_reply_to_user_id == tweet.user.id]

        print("q='to:{}'".format(tweet.user.screen_name))
        print("Found {} results, with final {}".format(len(raw_search_results), len(filtered_search_results)))

        # Construct the tree for this tweet
        new_reply_node = Node(tweet)

        # Base case is when there are no found replies to the given tweet
        for reply_tweet in filtered_search_results:
            new_reply_node.add_child(self.construct_reply_tree(reply_tweet))

        return new_reply_node

    def construct_reply_thread(self, tweet):
        reply_thread_root = self.find_root_ancestor(tweet)
        reply_thread_tree = self.construct_reply_tree(reply_thread_root)
        return reply_thread_tree

