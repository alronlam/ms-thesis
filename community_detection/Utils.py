def remove_communities_with_less_than_n(membership, n):
    return [m for m in membership if membership.count(m) > n ]


def construct_graph_with_filtered_communities(g, membership, min_vertices_per_community):
    g = g.copy()
    valid_membership = remove_communities_with_less_than_n(membership, 10)
    to_delete_ids = [v.index for v in g.vs if membership[v.index] not in valid_membership]
    g.delete_vertices(to_delete_ids)

    return (g, valid_membership)