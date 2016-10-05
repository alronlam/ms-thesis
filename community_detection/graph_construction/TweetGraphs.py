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


def construct_user_graph(graph, tweet_ids, pickle_file_name, limit=10000, start_index=0):
    if graph is None:
        graph = Graph(directed=True)

    new_edges = set()
    found_tweets = 0
    for index, tweet_id in enumerate(tweet_ids):

        if index >= start_index:

            print("Processing {}/{}".format(found_tweets, index))

            tweet = DBManager.get_or_add_tweet(tweet_id)

            if tweet is not None:
                found_tweets += 1
                user_id = tweet.user.id_str
                username = tweet.user.screen_name

                print("Processing tweet id {} posted by {} with id {}".format(tweet_id, username, user_id))

                add_user_vertex(graph, user_id, username)

                # construct directed edges if user A follows user B
                # loop through all vertices and check if they are in following or followers then create appropriate edge
                all_user_ids = graph.vs["id"]

                # this code is flawed because DBManager.get followers/following should be corrected. they do not get all the followers/following due to pagination
                follower_ids = DBManager.get_or_add_followers_ids(user_id)
                following_ids = DBManager.get_or_add_following_ids(user_id)

                for other_user_id in all_user_ids:
                    if follower_ids and other_user_id in follower_ids:
                        new_edges.add((other_user_id, user_id))

                    if following_ids and other_user_id in following_ids:
                        new_edges.add((user_id, other_user_id))

                # for other_user_id in all_user_ids:
                #     friendship = DBManager.get_or_add_friendship(user_id, other_user_id)
                #
                #     if friendship:
                #         if user_id < other_user_id:
                #             if friendship["following"] is True:
                #                 new_edges.add((user_id, other_user_id))
                #             if friendship["followed_by"] is True:
                #                 new_edges.add((other_user_id, user_id))
                #
                #         else:
                #             if friendship["following"] is True:
                #                 new_edges.add((other_user_id, user_id))
                #             if friendship["followed_by"] is True:
                #                 new_edges.add((user_id, other_user_id))


                graph.add_edges(list(new_edges))
                graph.save(pickle_file_name)
                new_edges = set()
                print("Saved {} at tweet index {}".format(pickle_file_name, index))
                print("# of edges and vertices after processing {} - {} - {}".format(user_id, new_edges.__len__(), all_user_ids.__len__()))
                print()

    # print("Final edges to be added: ")
    # print(new_edges)
    # graph.add_edges(list(new_edges))

    return graph


def add_user_vertex(graph, user_id, username):
    if not exists_in_graph(graph, user_id):
        new_vertex = graph.add_vertex(str(user_id))
        graph.vs[graph.vcount() - 1]["username"] = username
        graph.vs[graph.vcount() - 1]["id"] = str(user_id)
        graph.vs[graph.vcount() - 1]["name"] = str(user_id)

    return graph


def add_tweet_vertex(graph, tweet_id):
    if not exists_in_graph(graph, tweet_id):
        new_vertex = graph.add_vertex(str(tweet_id))
        new_tweet = DBManager.get_or_add_tweet(tweet_id)
        if new_tweet is not None:
            graph.vs[graph.vcount() - 1]["text"] = new_tweet.text
            graph.vs[graph.vcount() - 1]["tweet_id"] = new_tweet.id

    return graph


def exists_in_graph(graph, id):
    return graph.vcount() > 0 and graph.vs.select(name=str(id)).__len__() > 0
