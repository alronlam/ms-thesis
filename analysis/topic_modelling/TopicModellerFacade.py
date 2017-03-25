from community_detection.Utils import get_communities, get_vertex_ids_in_each_community, \
    get_tweet_texts_belonging_to_user_ids, get_user_ids_from_vertex_ids
from sentiment_analysis.preprocessing import PreProcessing


def construct_topic_models_for_communities(topic_modeller, graph, membership, tweet_objects, preprocessors=[]):

    community_topics_tuple_list = []

    vertex_ids_per_community = get_vertex_ids_in_each_community(graph, membership)

    for community_num, vertex_ids in enumerate(vertex_ids_per_community):
        user_ids_str = get_user_ids_from_vertex_ids(graph, vertex_ids)
        tweet_texts = get_tweet_texts_belonging_to_user_ids(tweet_objects, user_ids_str)
        tweet_texts = PreProcessing.preprocess_strings(tweet_texts, preprocessors)
        community_topics_tuple_list.append((community_num, topic_modeller.model_topics(tweet_texts)))

    return community_topics_tuple_list