from tweets import TweepyHelper
from data_structures.Node import Node
from functools import reduce
from database import DBManager
from igraph import *

class TweetUtils:

    def __init__(self):
        self.api = TweepyHelper.api

    # Methods for constructing reply threads/tweets
    def list_reply_ancestors(self, tweet):

        ancestor_id = tweet.in_reply_to_status_id
        ancestor_tweet = TweepyHelper.retrieve_tweet(ancestor_id)

        if ancestor_tweet is None:
            return [tweet]
        else:
            return self.list_reply_ancestors(ancestor_tweet) + [tweet]

    def find_root_ancestor(self, tweet):
        ancestor_id = tweet.in_reply_to_status_id
        ancestor_tweet = TweepyHelper.retrieve_tweet(ancestor_id)

        if ancestor_tweet is None:
            return tweet
        else:
            return self.find_root_ancestor(ancestor_tweet)

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

    # Methods for counting replies
    def count_replies_list(self, tweets):
        replies_list = []
        tweets_processed = 0
        for tweet in tweets:
            status = TweepyHelper.retrieve_tweet(tweet['id'])
            if status is not None:
                replies_list.append({'tweet_id': tweet['id'], 'reply_length': self.list_reply_ancestors(status).__len__()})

            tweets_processed += 1
            if tweets_processed % 10 == 0:
                print("Tweets Processed: {}\n".format(tweets_processed))

        return replies_list

    def count_replies_generator(self, tweets):
        for tweet in tweets:
            yield {'tweet_id': tweet.id, 'reply_length': self.list_reply_ancestors(tweet).__len__()}

    def reduce_max_reply_length(self, reply_lengths):
        # This only returns one max even if there are multiple max values
        try:
            return reduce(lambda x, y: x if x['reply_length'] > y['reply_length'] else y, reply_lengths)
        except:
            return reply_lengths

    def frequency_count_reply_lengths(self, reply_lengths):
        frequency_count = {}
        for reply_length in reply_lengths:
            actual_length = reply_length['reply_length']
            frequency_count[actual_length] = frequency_count.get(actual_length, 0) + 1
        return frequency_count

    def filter_reply_lengths_gte(self, reply_lengths, min_length):
        return (reply_length for reply_length in reply_lengths if reply_length['reply_length']  >= min_length)


    # Methods for constructing the graph
    def construct_follow_graph(self, graph, user_id, vertices_limit, is_directed):

        if graph is None:
            graph = Graph()

        if graph.vcount() >= vertices_limit:
            return graph

        graph = self.add_vertex(graph, user_id) # Add user_id as initial vertex

        # Get info regarding following/followers
        following_ids = DBManager.get_or_add_following_ids(user_id)
        followers_ids = DBManager.get_or_add_followers_ids(user_id)

        if following_ids is not None and followers_ids is not None:

            intersection_ids = [id for id in followers_ids if id in following_ids]

            # Add appropriate vertices and edges

            if is_directed:
                pass # stub
            else:

                for intersection_id in intersection_ids[0:max(vertices_limit-graph.vcount(), 0)]:
                    graph = self.add_vertex(graph, intersection_id)
                    graph.add_edge(str(user_id), str(intersection_id))

                for intersection_id in intersection_ids:
                    graph = self.construct_follow_graph(graph, intersection_id, vertices_limit, is_directed)

        return graph


    def add_vertex(self, graph, user_id):
        if graph.vcount() == 0 or graph.vs.select(name = str(user_id)).__len__() == 0:
            new_vertex = graph.add_vertex(str(user_id))
            new_user = DBManager.get_or_add_user(user_id)
            if new_user is not None:
                graph.vs[graph.vcount()-1]["screen_name"] = new_user.screen_name

        return graph