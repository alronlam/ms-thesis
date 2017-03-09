import pickle
from collections import Counter

from igraph import Graph

from community_detection.graph_construction.TweetGraphs import add_user_vertex


def construct_user_mention_hashtag_sa_graph(graph, tweets, classifier, pickle_file_name, start_index=0, verbose=False):

    if graph is None:
        graph = Graph(directed=False)

    new_edges = set()

    for index, tweet in enumerate(tweets):
        user_id_str = tweet.user.id_str
        user_screen_name = tweet.user.screen_name

        ### CREATE VERTICES ###
        add_user_vertex(graph, user_id_str, user_screen_name)
        if verbose:
            print("Constructing user mention hashtag SA graph: processed {}/{} tweets".format(index, len(tweets)))

    ### CREATE EDGES ###
    if verbose:
        print("Constructing mention scores")
    mention_scores = score_mentions(tweets)
    if verbose:
        print("Mention scores length: {}".format(len(mention_scores)))

    if verbose:
        print("Constructing hashtag scores")
    hashtag_scores = score_hashtags(tweets)
    if verbose:
        print("Hashtag scores length: {}".format(len(hashtag_scores)))

    if verbose:
        print("Constructing sa scores")
    sa_scores = score_sa(tweets, classifier)
    if verbose:
        print("SA scores length: {}".format(len(sa_scores)))

    if verbose:
        print("Consolidating scores")
    final_scores = consolidate([mention_scores, hashtag_scores, sa_scores])
    if verbose:
        print("Finalize scores length: {}".format(len(final_scores)))
    # print(Counter([score for key, score in final_scores.items()]))

    # if verbose:
    #     print("Normalizing scores")
    # final_scores = normalize(final_scores)
    # print(Counter([score for key, score in final_scores.items()]))

    graph.save("{}-without-edges".format(pickle_file_name))
    pickle.dump(final_scores, open("{}.finalscores".format(pickle_file_name), "rb"))

    THRESHOLD = 3

    if verbose:
        print("Adding edges based on threshold score")

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
    graph.save(pickle_file_name)

    return graph

# {(a,b): }
def construct_ordered_tuple(a, b):
    if a <= b:
        return (a,b)
    return construct_ordered_tuple(b,a)

def score_mentions(tweets):
    score_dict = {}

    for index, tweet in enumerate(tweets):
        user_id_str = tweet.user.id_str

        mentions_idstr_screenname_tuples = [(mention_dict["id_str"], mention_dict["screen_name"]) for mention_dict in tweet.entities.get('user_mentions')]
        for other_user_id_str, other_user_screen_name in mentions_idstr_screenname_tuples:
            ordered_tuple = construct_ordered_tuple(user_id_str, other_user_id_str)
            score_dict[ordered_tuple] = score_dict.get(ordered_tuple, 0) + 1

    return score_dict

def score_hashtags(tweets):

    # group users according to hashtag
    hashtag_users_dict = {}

    for index, tweet in enumerate(tweets):
        user_id_str = tweet.user.id_str
        hashtags = [hashtag_dict["text"].lower() for hashtag_dict in tweet.entities.get('hashtags')]

        for hashtag in hashtags:
            hashtag_user_set = hashtag_users_dict.get(hashtag, set())
            hashtag_user_set.add(user_id_str)
            hashtag_users_dict[hashtag] = hashtag_user_set

    # construct score dict
    score_dict = {}
    for hashtag, user_set in hashtag_users_dict.items():
        user_list = list(user_set)
        for index, user in enumerate(user_list):
            for other_user in user_list[index+1:]:
                ordered_tuple = construct_ordered_tuple(user, other_user)
                score_dict[ordered_tuple] = score_dict.get(ordered_tuple, 0) + 1

    return score_dict

def score_sa(tweets, classifier):
    # group users according to hashtag
    hashtag_users_dict = {}

    for index, tweet in enumerate(tweets):
        user_id_str = tweet.user.id_str
        hashtags = [hashtag_dict["text"].lower() for hashtag_dict in tweet.entities.get('hashtags')]
        sentiment = classifier.classify_sentiment(tweet.text, {})

        for hashtag in hashtags:
            hashtag_user_set = hashtag_users_dict.get((hashtag,sentiment), set())
            hashtag_user_set.add(user_id_str)
            hashtag_users_dict[(hashtag,sentiment)] = hashtag_user_set

    # construct score dict
    score_dict = {}
    for (hashtag, sentiment), user_set in hashtag_users_dict.items():
        user_list = list(user_set)
        for index, user in enumerate(user_list):
            for other_user in user_list[index+1:]:
                ordered_tuple = construct_ordered_tuple(user, other_user)
                score_dict[ordered_tuple] = score_dict.get(ordered_tuple, 0) + 1

    return score_dict

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

