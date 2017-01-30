def count_mutual_edges(graph):
    return sum([1 for boolean_result in graph.is_mutual(graph.es) if boolean_result])/2