import pickle


original_base_graph_names = ["brexit_mention_graph.pickle",
                             "brexit_mention_graph_with_hashtags.pickle",
                             "brexit_mention_graph_with_hashtags_contextualsa.pickle",
                             "brexit_mention_graph_with_hashtags_sa.pickle",
                             "threshold-0.04-brexit_mention_hashtag_contextualsa_graph.pickle",
                             "threshold-0.04-brexit_mention_hashtag_sa_graph.pickle",
                             "threshold-0.05-brexit_mention_hashtag_contextualsa_graph.pickle",
                             "threshold-0.05-brexit_mention_hashtag_sa_graph.pickle"
]

modified_base_graph_names = [ "300-brexit_mention_graph.pickle",
                     "300-brexit_mention_graph_with_hashtags.pickle",
                     "300-brexit_mention_graph_with_hashtags_contextualsa.pickle",
                     "300-brexit_mention_graph_with_hashtags_sa.pickle",
                    "100-threshold-0.04-brexit_mention_hashtag_contextualsa_graph.pickle",
                    "100-threshold-0.04-brexit_mention_hashtag_sa_graph.pickle",
                    "100-threshold-0.05-brexit_mention_hashtag_contextualsa_graph.pickle",
                    "100-threshold-0.05-brexit_mention_hashtag_sa_graph.pickle",]

for base_graph_name in original_base_graph_names:

    graph = pickle.load(open("community_detection/graph_info/{}".format(base_graph_name), "rb"))
    membership = pickle.load(open("community_detection/graph_info/{}.membership".format(base_graph_name), "rb"))

    modularity = graph.modularity(membership)
    num_vertices = len(graph.vs)
    num_edges = len(graph.es)

    print("Modularity:{}\nNum Vertices:{}\nNum Edges: {}\n".format(modularity, num_vertices, num_edges), file=open("community_detection/graph_info/{}_general_info.txt".format(base_graph_name),"w"))