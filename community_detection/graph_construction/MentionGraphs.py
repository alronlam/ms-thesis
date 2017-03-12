import pickle
from collections import Counter

from igraph import Graph

from community_detection.graph_construction.TweetGraphs import add_user_vertex
from sentiment_analysis.preprocessing import PreProcessing
from twitter_data.database import DBUtils


def construct_user_mention_hashtag_sa_graph(graph, tweets, classifier, pickle_file_name, THRESHOLD=0.5, hashtag_preprocessors=[], sa_preprocessors=[], verbose=False, load_mode=False):

    if load_mode:
        graph = pickle.load(open("without-edges-{}".format(pickle_file_name), "rb"))
        final_scores = pickle.load(open("final-scores-{}".format(pickle_file_name), "rb"))
    else:
        if graph is None:
            graph = Graph(directed=False)

        for index, tweet in enumerate(tweets):
            user_id_str = tweet.user.id_str
            user_screen_name = tweet.user.screen_name

            ### CREATE VERTICES ###
            add_user_vertex(graph, user_id_str, user_screen_name)
            if verbose:
                if index % 1000 == 0 or index == len(tweets)-1:
                    print("Constructing user mention hashtag SA graph: processed {}/{} tweets".format(index+1, len(tweets)))

        ### CREATE EDGES ###
        if verbose:
            print("Constructing mention scores")
        mention_scores = score_mentions(tweets, graph)
        if verbose:
            print("Mention scores length: {}".format(len(mention_scores)))


        if verbose:
            print("Constructing hashtag scores")

        preprocessed_tweets_for_hashtags = PreProcessing.preprocess_tweets(tweets, hashtag_preprocessors)
        #extracting this after preprocessing to remove the universal hashtag(s)
        unique_hashtags = get_unique_hashtags(preprocessed_tweets_for_hashtags)

        mention_hashtag_scores = score_hashtags_optimized(preprocessed_tweets_for_hashtags, mention_scores, unique_hashtags)

        if verbose:
            print("Constructing sa scores")
        preprocessed_tweets_for_sa = PreProcessing.preprocess_tweets(tweets, sa_preprocessors)
        mention_hashtag_sa_scores = score_sa_optimized(preprocessed_tweets_for_sa, classifier, mention_hashtag_scores, unique_hashtags)

        if verbose:
            print("Normalizing scores")
        final_scores = normalize(mention_hashtag_sa_scores)
        if verbose:
            print(Counter([score for key, score in final_scores.items()]))

        graph.save("without-edges-{}".format(pickle_file_name))
        pickle.dump(final_scores, open("final-scores-{}".format(pickle_file_name), "wb"))

    if verbose:
        print("Creating list of edges based on threshold score")

    new_edges = set()
    count = 0
    for tuple, score in final_scores.items():
        if score >= THRESHOLD:
            new_edges.add(tuple)

        count += 1
        # print("Adding edge: Processed {}/{} scores.".format(count, len(final_scores.items())))

    if verbose:
        print("Adding {} edges".format(len(new_edges)))
    graph.add_edges(list(new_edges))
    graph.es["weight"] = 1
    graph.save("threshold-{}-{}".format(THRESHOLD, pickle_file_name))

    return graph

# {(a,b): }
def construct_ordered_tuple(a, b):
    if a <= b:
        return (a,b)
    return construct_ordered_tuple(b,a)

def score_mentions(tweets, graph):
    score_dict = {}

    for index, tweet in enumerate(tweets):
        user_id_str = tweet.user.id_str

        mentions_idstr_screenname_tuples = [(mention_dict["id_str"], mention_dict["screen_name"]) for mention_dict in tweet.entities.get('user_mentions')]
        for other_user_id_str, other_user_screen_name in mentions_idstr_screenname_tuples:
            add_user_vertex(graph, other_user_id_str, other_user_screen_name)
            ordered_tuple = construct_ordered_tuple(user_id_str, other_user_id_str)
            score_dict[ordered_tuple] = score_dict.get(ordered_tuple, 0) + 1

    return score_dict

def score_hashtags_optimized(tweets, score_dict, unique_hashtags):
    if not unique_hashtags:
        unique_hashtags = get_unique_hashtags(tweets)

    for hashtag in unique_hashtags:
        user_set = set()
        for tweet in tweets:
            curr_user = tweet.user.id_str
            if hashtag in [hashtag_dict["text"].lower() for hashtag_dict in tweet.entities.get('hashtags')]:
                for other_user in user_set:
                    tuple = construct_ordered_tuple(curr_user, other_user)
                    if tuple in score_dict: # only consider those entries already present in the score dict
                        score_dict[tuple] += 1
                user_set.add(curr_user)
        del user_set
    return score_dict

def score_hashtags(tweets, unique_hashtags=None):

    if not unique_hashtags:
        unique_hashtags = get_unique_hashtags(tweets)

    score_dict = {}

    for hashtag in unique_hashtags:
        user_set = set()
        for tweet in tweets:
            curr_user = tweet.user.id_str
            if hashtag in [hashtag_dict["text"].lower() for hashtag_dict in tweet.entities.get('hashtags')]:
                for other_user in user_set:
                    tuple = construct_ordered_tuple(curr_user, other_user)
                    score_dict[tuple] = score_dict.get(tuple, 0) + 1
                user_set.add(curr_user)
        del user_set
    return score_dict


def score_sa_optimized(tweets, classifier, score_dict, unique_hashtags):
    for hashtag in unique_hashtags:
        positive_user_set = set()
        negative_user_set = set()
        for tweet in tweets:
            curr_user = tweet.user.id_str
            if hashtag in [hashtag_dict["text"].lower() for hashtag_dict in tweet.entities.get('hashtags')]:

                sentiment = classifier.classify_sentiment(tweet.text, DBUtils.retrieve_full_conversation(tweet.in_reply_to_status_id, []))

                user_set = None
                if sentiment == 'positive':
                    user_set = positive_user_set
                elif sentiment == 'negative':
                    user_set = negative_user_set

                if user_set:
                    for other_user in user_set:
                        tuple = construct_ordered_tuple(curr_user, other_user)
                        if tuple in score_dict:
                            score_dict[tuple] += 1
                    user_set.add(curr_user)

        del positive_user_set
        del negative_user_set
    return score_dict

def score_sa(tweets, classifier, unique_hashtags=None):

    if not unique_hashtags:
        unique_hashtags = get_unique_hashtags(tweets)

    score_dict = {}

    for hashtag in unique_hashtags:
        positive_user_set = set()
        negative_user_set = set()
        for tweet in tweets:
            curr_user = tweet.user.id_str
            if hashtag in [hashtag_dict["text"].lower() for hashtag_dict in tweet.entities.get('hashtags')]:

                sentiment = classifier.classify_sentiment(tweet.text, {})

                user_set = None
                if sentiment == 'positive':
                    user_set = positive_user_set
                elif sentiment == 'negative':
                    user_set = negative_user_set

                if user_set:
                    for other_user in user_set:
                        tuple = construct_ordered_tuple(curr_user, other_user)
                        score_dict[tuple] = score_dict.get(tuple, 0) + 1
                    user_set.add(curr_user)

        del positive_user_set
        del negative_user_set
    return score_dict

def get_unique_hashtags(tweets):
    unique_hashtags = set()
    for tweet in tweets:
        tweet_hashtags = [hashtag_dict["text"].lower() for hashtag_dict in tweet.entities.get('hashtags')]
        for tweet_hashtag in tweet_hashtags:
            unique_hashtags.add(tweet_hashtag)
    return unique_hashtags

# def score_sa(tweets, classifier):
#     # group users according to hashtag
#     hashtag_users_dict = {}
#
#     for index, tweet in enumerate(tweets):
#         user_id_str = tweet.user.id_str
#         hashtags = [hashtag_dict["text"].lower() for hashtag_dict in tweet.entities.get('hashtags')]
#         sentiment = classifier.classify_sentiment(tweet.text, {})
#
#         for hashtag in hashtags:
#             hashtag_user_set = hashtag_users_dict.get((hashtag,sentiment), set())
#             hashtag_user_set.add(user_id_str)
#             hashtag_users_dict[(hashtag,sentiment)] = hashtag_user_set
#
#     # construct score dict
#     score_dict = {}
#     for (hashtag, sentiment), user_set in hashtag_users_dict.items():
#         user_list = list(user_set)
#         for index, user in enumerate(user_list):
#             for other_user in user_list[index+1:]:
#                 ordered_tuple = construct_ordered_tuple(user, other_user)
#                 score_dict[ordered_tuple] = score_dict.get(ordered_tuple, 0) + 1
#
#     return score_dict

def consolidate(score_dict_list):
    final_dict = {}
    for score_dict in score_dict_list:
        for tuple, score in score_dict.items():
            final_dict[tuple] = final_dict.get(tuple, 0) + score

    return final_dict

def normalize(score_dict):
    max_score = get_max_score(score_dict)
    for tuple, score in score_dict.items():
        score_dict[tuple] = score/max_score

    print("Mention graph normalization max score: {}".format(max_score))

    return score_dict

def get_max_score(score_dict):
    return max([score for tuple, score in score_dict.items()])

def calculate_total_score(score_dict):
    return sum([score for tuple, score in score_dict.items()])

