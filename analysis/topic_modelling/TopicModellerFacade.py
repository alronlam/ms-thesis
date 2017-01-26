from community_detection.Utils import get_communities, get_vertex_ids_in_each_community, \
    filter_tweets_belonging_to_user_ids, get_user_ids_from_vertex_ids


def construct_topic_models_for_communities(topic_modeller, graph, membership, tweet_objects):

    vertex_ids_per_community = get_vertex_ids_in_each_community(graph, membership)

    for community_num, vertex_ids in enumerate(vertex_ids_per_community):
        user_ids_str = get_user_ids_from_vertex_ids(graph, vertex_ids)
        tweet_texts = filter_tweets_belonging_to_user_ids(tweet_objects, user_ids_str)

        print("Community # {} - {}".format(community_num, topic_modeller.model_topics(tweet_texts)))








