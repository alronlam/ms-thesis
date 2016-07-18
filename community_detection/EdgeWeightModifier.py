import abc

class EdgeWeightModifierBase(object):

    @abc.abstractmethod
    def modify_edge_weights(self, graph):
        """
        :param graph: graph object to be modified
        :return: graph with modified edge weights
        """

class SAWeightModifier(EdgeWeightModifierBase):
    def modify_edge_weights(self, graph):
        return graph


def modify_edge_weights(graph, edge_weight_modifiers):
    for modifier in edge_weight_modifiers:
        graph = modifier.modify_edge_weights(graph)
    return graph