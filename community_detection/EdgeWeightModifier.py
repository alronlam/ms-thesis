import abc

class EdgeWeightModifierBase(object):

    @abc.abstractmethod
    def modify_edge_weights(self, graph):
        """
        :param graph: graph object to be modified
        :return: graph with modified edge weights
        """

class TweetVerticesSAWeightModifier(EdgeWeightModifierBase):

    def __init__(self, sentiment_classifier):
        self.classifier = sentiment_classifier

    def modify_edge_weights(self, graph):
        sentiment_dict = {}
        for edge in graph.es:
            source_vertex_id = edge.source
            target_vertex_id = edge.target
            source_vertex = graph.vs[source_vertex_id]
            target_vertex = graph.vs[target_vertex_id]

            source_vertex_sentiment = self.get_or_add_sentiment(source_vertex, sentiment_dict)
            target_vertex_sentiment = self.get_or_add_sentiment(target_vertex, sentiment_dict)

            if source_vertex_sentiment == target_vertex_sentiment:
                edge["weight"] += 1000

        return graph

    def get_or_add_sentiment(self, vertex, sentiment_dict):
        sentiment = sentiment_dict.get(vertex["tweet_id"], None)
        if not sentiment:
            sentiment = self.classifier.classify_sentiment(vertex["text"])
            sentiment_dict[vertex["tweet_id"]] = sentiment
            vertex["sentiment"] = sentiment
            # vertex["text"] = "(" + sentiment + ") " + vertex["text"]
        return sentiment

def modify_edge_weights(graph, edge_weight_modifiers):
    for modifier in edge_weight_modifiers:
        graph = modifier.modify_edge_weights(graph)
    return graph