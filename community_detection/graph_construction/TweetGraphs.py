from database import DBManager
from igraph import *
from tweets import TweepyHelper

# edge exists if tweet_graph has same hashtag
def construct_tweet_graph(graph, tweets, limit=10000, start_index=0):

    if graph is None:
        graph = Graph()

    hashtag_dict = {}
    for index, tweet in enumerate(tweets):
        print("Processed {}/{}".format(index, limit))
        if index >= start_index:
            add_tweet_vertex(graph, tweet.id)
            hashtags = tweet.entities.get('hashtags')

            for hashtag in hashtags:
                tweet_id_list = hashtag_dict.get(hashtag["text"], [])

                for other_tweet_id in tweet_id_list:
                     graph.add_edge(str(tweet.id), str(other_tweet_id), weight=1)

                tweet_id_list.append(tweet.id)
                hashtag_dict[hashtag["text"]] = tweet_id_list

            if index % 10 == 0:
                graph.save("2016-03-04-tweets-pilipinasdebates.pickle")

        if index == limit:
            break
    return graph

def construct_user_graph(graph, tweet_ids, limit=10000, start_index=0):
    if graph is None:
        graph = Graph(directed=True)

    for index, tweet_id in enumerate(tweet_ids):

        print("Processed {}".format(index))

        tweet = DBManager.get_or_add_tweet(tweet_id)

        if tweet is not None:
            print("Found tweet")
            # user_id = tweet["user"]["id_str"]
            user_id = tweet.user.id
            username = tweet.user.screen_name

            add_user_vertex(graph, user_id, username)

            # construct directed edges if user A follows user B
            # loop through all vertices and check if they are in following or followers then create appropriate edge
            all_user_ids = graph.vs["id"]
            new_edges = []
            for other_user_id in all_user_ids:
                if other_user_id != user_id:
                    friendship_result = TweepyHelper.show_friendship(user_id, other_user_id)
                    if friendship_result[0]['following'] is True:
                        new_edges.append((user_id, other_user_id))
                    if friendship_result[0]['followed_by'] is True:
                        new_edges.append((other_user_id, user_id))

            graph.add_edges(new_edges)

    return graph



def add_user_vertex(graph, user_id, username):
    if not exists_in_graph(graph, user_id):
        new_vertex = graph.add_vertex(str(user_id))
        graph.vs[graph.vcount()-1]["username"] = username
        graph.vs[graph.vcount()-1]["id"] = user_id
        graph.vs[graph.vcount()-1]["name"] = username


    return graph

def add_tweet_vertex(graph, tweet_id):
    if not exists_in_graph(graph, tweet_id):
        new_vertex = graph.add_vertex(str(tweet_id))
        new_tweet = DBManager.get_or_add_tweet(tweet_id)
        if new_tweet is not None:
            graph.vs[graph.vcount()-1]["text"] = new_tweet.text
            graph.vs[graph.vcount()-1]["tweet_id"] = new_tweet.id

    return graph

def exists_in_graph(graph, id):
    return graph.vcount() > 0 and graph.vs.select(name = str(id)).__len__() > 0