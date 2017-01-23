from community_detection.weight_modification.EdgeWeightModifier import EdgeWeightModifierBase


class UserVerticesMentionsWeightModifier(EdgeWeightModifierBase):

    def modify_edge_weights(self, graph, params, verbose):

        tweets = params["tweets"]

        total_weight_update = 0
        for index, tweet in enumerate(tweets):
            user_id_str = tweet.user.id_str
            # in_reply_to_user_id_str = tweet.in_reply_to_user_id_str # not sure if this is needed
            mentioned_user_ids = [user_mention_dict["id_str"] for user_mention_dict in tweet.entities.get('user_mentions')]

            for mentioned_user_id in mentioned_user_ids:

                try:
                    this_user_v_index = graph.vs.find(name=user_id_str).index
                    other_user_v_index = graph.vs.find(name=mentioned_user_id).index

                    edge1 = graph.es.find(_source=this_user_v_index, _target=other_user_v_index)
                    edge2 = graph.es.find(_source=other_user_v_index, _target=this_user_v_index)

                    edge1["weight"] += 1
                    edge2["weight"] += 1

                    total_weight_update += 1
                except Exception as e:
                    if verbose:
                        print(e)

            if verbose:
                print("UserVerticesMentionsWeightModifier: Processed {}/{} tweets".format(index+1, len(tweets)))

        if verbose:
            print("Total number of edges updated: {}".format(total_weight_update))
        return graph