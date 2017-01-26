def remove_communities_with_less_than_n(membership, n):
    return [m for m in membership if membership.count(m) > n ]


def construct_graph_with_filtered_communities(g, membership, min_vertices_per_community):
    g = g.copy()
    valid_membership = remove_communities_with_less_than_n(membership, 10)
    to_delete_ids = [v.index for v in g.vs if membership[v.index] not in valid_membership]
    g.delete_vertices(to_delete_ids)

    return (g, valid_membership)

def get_communities(membership):
    return sorted(list(set(membership)))

def get_vertex_ids_in_community(graph, membership, community_num):
    return [ index for index, x in enumerate(membership) if x == community_num]

def get_vertex_ids_in_each_community(graph, membership):
    communities = get_communities(membership)
    community_vertices = []
    for index, community in enumerate(communities):
        community_vertices.append(get_vertex_ids_in_community(graph, membership, index))
    return community_vertices

def get_user_ids_from_vertex_ids(graph, vertex_ids):
    return [vertex["name"] for vertex in graph.vs if vertex.index in vertex_ids]

def filter_tweets_belonging_to_user_ids(tweet_objects, user_ids_str):
    return [tweet.text for tweet in tweet_objects if tweet.user.id_str in user_ids_str]
