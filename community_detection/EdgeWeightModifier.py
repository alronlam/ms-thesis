import abc

class EdgeWeightModifierBase(object):

    @abc.abstractmethod
    def modify_edge_weights(self, graph, params):
        """
        :param graph: graph object to be modified
        :return: graph with modified edge weights
        """

class UserVerticesSAWeightModifier(EdgeWeightModifierBase):
    def __init__(self, sentiment_classifier):
        self.classifier = sentiment_classifier

    def modify_edge_weights(self, graph, params):
        hashtag_sentiment_users_dict = {}
        tweets = params["tweets"]
        contextual_info_dict = params.get("contextual_info")

        for tweet in tweets:
            user_id_str = tweet.user.id_str
            sentiment = self.classifier.classify_sentiment(tweet.text, contextual_info_dict)
            hashtags = [hashtag_dict["text"].lower() for hashtag_dict in tweet.entities.get('hashtags')]

            for hashtag in hashtags:
                hashtag_sentiment_set = self.get_or_add_hashtag_sentiment_set(hashtag_sentiment_users_dict, hashtag, sentiment )
                hashtag_sentiment_set.add(user_id_str)

        for key, user_set in hashtag_sentiment_users_dict.items():
            print("{} - {}".format(key, len(user_set)))

            user_list = list(user_set)

            for index in range(len(user_list)-1):
                user1 = user_list[index]

                for index2 in range(index, len(user_list)):
                    user2 = user_list[index2]

                    if user1 != user2:
                        try:
                            edge12 = graph.es[graph.get_eid(user1, user2)]
                            edge12["weight"] += 1
                            print("New edge weight of {} to {} is {}".format(user1, user2, edge12["weight"]))
                        except Exception as e:
                            # print(e)
                            pass

                        try:
                            edge21 = graph.es[graph.get_eid(user2, user1)]
                            edge21["weight"] += 1
                            print("New edge weight of {} to {} is {}".format(user2, user1, edge21["weight"]))
                        except:
                            pass


        return graph

    def get_or_add_hashtag_sentiment_set(self, hashtag_sentiment_users_dict, hashtag, sentiment):
        hashtag_sentiment_set = hashtag_sentiment_users_dict.get((hashtag, sentiment), None)
        if not hashtag_sentiment_set:
            hashtag_sentiment_set = set()
            hashtag_sentiment_users_dict[(hashtag, sentiment)] = hashtag_sentiment_set

        return hashtag_sentiment_set






class TweetVerticesSAWeightModifier(EdgeWeightModifierBase):

    def __init__(self, sentiment_classifier):
        self.classifier = sentiment_classifier

    def modify_edge_weights(self, graph, params):
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

def modify_edge_weights(graph, edge_weight_modifiers, params):
    for modifier in edge_weight_modifiers:
        graph = modifier.modify_edge_weights(graph, params)
    return graph