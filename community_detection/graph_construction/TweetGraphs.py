from igraph import *

from twitter_data.database import DBManager


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


def construct_tweet_hashtag_graph_with_sentiment(graph, tweets, pickle_file_name, sentiment_classifier):
    if graph is None:
        graph = Graph(directed=False)

    new_edges = set()

    for index, tweet in enumerate(tweets):
        add_tweet_vertex(graph, tweet)
        hashtags = [hashtag_dict["text"].lower() for hashtag_dict in tweet.entities.get('hashtags')]

        sentiment = sentiment_classifier.classify_sentiment(tweet.text, {})

        hashtag_sentiment_set = set()

        for hashtag in hashtags:
            add_hashtag_vertex(graph, hashtag+"-"+sentiment)
            hashtag_sentiment_set.add(hashtag+"-"+sentiment)

        # edges
        # USER TO HASHTAG EDGE
        for hashtag in hashtag_sentiment_set:
            new_edges.add((str(tweet.id), hashtag))

        print("Constructing base graph: Processed {}/{} tweets.".format(index,len(tweets)))

    graph.add_edges(list(new_edges))
    graph.save(pickle_file_name)

    return graph


def construct_user_graph(graph, tweet_objects, pickle_file_name, limit=10000, start_index=0, verbose=False):
    if graph is None:
        graph = Graph(directed=True)

    new_edges = set()
    found_tweets = 0
    for index, tweet_object in enumerate(tweet_objects):

        if index >= start_index:

            found_tweets += 1
            user_id = tweet_object.user.id_str
            username = tweet_object.user.screen_name

            add_user_vertex(graph, user_id, username)

            # construct directed edges if user A follows user B
            # loop through all vertices and check if they are in following or followers then create appropriate edge
            all_user_ids = graph.vs["name"]

            # this code is flawed because DBManager.get followers/following should be corrected. it currently has a limit to avoid being stuck with one user
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

            if verbose:
                # print("Saved {} at tweet index {}".format(pickle_file_name, index))
                print("# of edges and vertices after processing {} - {} - {}".format(user_id, graph.ecount(), all_user_ids.__len__()))

    graph.add_edges(list(new_edges))
    graph.es["weight"] = 1
    graph.save(pickle_file_name)
        # new_edges = set()


    # print("Final edges to be added: ")
    # print(new_edges)
    # graph.add_edges(list(new_edges))

    return graph


def construct_user_hashtag_graph(graph, tweets,  pickle_file_name, start_index=0, verbose=False):

    if graph is None:
        graph = Graph(directed=False)

    new_edges = set()

    for index, tweet in enumerate(tweets):
        user_id_str = tweet.user.id_str
        user_screen_name = tweet.user.screen_name

        hashtags = [hashtag_dict["text"].lower() for hashtag_dict in tweet.entities.get('hashtags')]

        ### CREATE VERTICES ###
        add_user_vertex(graph, user_id_str, user_screen_name)
        for hashtag in hashtags:
            add_hashtag_vertex(graph, hashtag)

        ### CREATE EDGES ###

        # USER TO HASHTAG EDGE
        for hashtag in hashtags:
            new_edges.add((user_id_str, hashtag))

        # USER TO USER EDGE
        all_vertex_ids = graph.vs["id"]

        # this code is flawed because DBManager.get followers/following should be corrected. it currently has a limit to avoid being stuck with one user
        follower_ids = DBManager.get_or_add_followers_ids(user_id_str)
        following_ids = DBManager.get_or_add_following_ids(user_id_str)

        for other_vertex_id in all_vertex_ids:
            if not other_vertex_id == user_id_str:
                if follower_ids and other_vertex_id in follower_ids:
                    new_edges.add((other_vertex_id, user_id_str))

                if following_ids and other_vertex_id in following_ids:
                    new_edges.add((user_id_str, other_vertex_id))

        graph.add_edges(list(new_edges))
        graph.save(pickle_file_name)
        new_edges = set()
        print("Saved {} at tweet index {}".format(pickle_file_name, index))
        print("Constructing base graph: Processed {}/{} tweets.".format(index,len(tweets)))
        print()

    return graph


def add_hashtag_vertex(graph, hashtag_text):
    if not exists_in_graph(graph, hashtag_text):
        new_vertex = graph.add_vertex(hashtag_text)
        graph.vs[graph.vcount()-1]["display_str"] = hashtag_text
        graph.vs[graph.vcount()-1]["id"] = hashtag_text
        graph.vs[graph.vcount()-1]["name"] = hashtag_text

    return graph

def add_user_vertex(graph, user_id, username):
    if not exists_in_graph(graph, user_id):
        new_vertex = graph.add_vertex(str(user_id))
        graph.vs[graph.vcount() - 1]["username"] = username
        graph.vs[graph.vcount() - 1]["display_str"] = username
        # graph.vs[graph.vcount() - 1]["id"] = str(user_id)
        graph.vs[graph.vcount() - 1]["name"] = str(user_id)

    return graph

def add_tweet_vertex(graph, tweet):
    if not exists_in_graph(graph, tweet.id):
        new_vertex = graph.add_vertex(str(tweet.id))
        graph.vs[graph.vcount() - 1]["tweet_text"] = tweet.text
        graph.vs[graph.vcount() - 1]["display_str"] = tweet.text
        graph.vs[graph.vcount() - 1]["tweet_id"] = tweet.id
    return graph

# def add_tweet_vertex(graph, tweet_id):
#     if not exists_in_graph(graph, tweet_id):
#         new_vertex = graph.add_vertex(str(tweet_id))
#         new_tweet = DBManager.get_or_add_tweet(tweet_id)
#         if new_tweet is not None:
#             graph.vs[graph.vcount() - 1]["text"] = new_tweet.text
#             graph.vs[graph.vcount() - 1]["tweet_id"] = new_tweet.id
#
#     return graph


def exists_in_graph(graph, id):
    return graph.vcount() > 0 and graph.vs.select(name=str(id)).__len__() > 0
