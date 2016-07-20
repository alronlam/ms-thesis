from database import DBManager
from igraph import *

# edge exists if tweet_graph has same hashtag
def construct_tweet_graph(graph, tweets, limit=10000, start_index=0):

    if graph is None:
        graph = Graph()

    hashtag_dict = {}
    for index, tweet in enumerate(tweets):
        print("Processed {}/{}".format(index, limit))
        if index >= start_index:
            add_vertex(graph, tweet.id)
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


def add_vertex(graph, tweet_id):
    if not user_exists_in_graph(graph, tweet_id):
        new_vertex = graph.add_vertex(str(tweet_id))
        new_tweet = DBManager.get_or_add_tweet(tweet_id)
        if new_tweet is not None:
            graph.vs[graph.vcount()-1]["text"] = new_tweet.text
            graph.vs[graph.vcount()-1]["tweet_id"] = new_tweet.id

    return graph

def user_exists_in_graph(graph, tweet_id):
    return graph.vcount() > 0 and graph.vs.select(name = str(tweet_id)).__len__() > 0