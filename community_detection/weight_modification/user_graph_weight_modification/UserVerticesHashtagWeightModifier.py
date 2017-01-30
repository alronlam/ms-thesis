from community_detection.weight_modification.EdgeWeightModifier import EdgeWeightModifierBase


class UserVerticesHashtagWeightModifier(EdgeWeightModifierBase):

    def modify_edge_weights(self, graph, params, verbose):
        hashtag_users_dict = {}

        tweets = params["tweets"]
        contextual_info_dict = params.get("contextual_info")

        if verbose:
            print("Constructing (hashtag, sentiment) -> user dictionary.")

        for index, tweet in enumerate(tweets):
            user_id_str = tweet.user.id_str
            hashtags = [hashtag_dict["text"].lower() for hashtag_dict in tweet.entities.get('hashtags')]

            for hashtag in hashtags:
                hashtag_user_set = self.get_or_add_hashtag_user_set(hashtag_users_dict, hashtag)
                hashtag_user_set.add(user_id_str)

            if verbose:
                print("UserVerticesHashtagWeightModifier: Processed {}/{} tweets".format(index+1, len(tweets)))

        if verbose:
            print("Going through edges.")

        total_weight_update = 0

        for index, edge in enumerate(graph.es):
            source_vertex_id = edge.source
            target_vertex_id = edge.target
            source_vertex_name = graph.vs[source_vertex_id]["name"]
            target_vertex_name = graph.vs[target_vertex_id]["name"]

            for key, user_set in hashtag_users_dict.items():
                if source_vertex_name in user_set and target_vertex_name in user_set:
                    edge["weight"] += 1
                    total_weight_update +=1
                    if verbose:
                        print("New edge weight of {} to {} is {}".format(source_vertex_name, target_vertex_name, edge["weight"]))

            if verbose:
                print("HashtagWeightModifier: Processed {}/{} edges".format(index+1, len(graph.es)))

        print("Hashtag: Modified edges {} times.".format(total_weight_update))
        return graph

    def get_or_add_hashtag_user_set(self, hashtag_users_dict, hashtag):
        hashtag_user_set = hashtag_users_dict.get(hashtag, None)
        if not hashtag_user_set:
            hashtag_user_set = set()
            hashtag_users_dict[hashtag] = hashtag_user_set

        return hashtag_user_set

