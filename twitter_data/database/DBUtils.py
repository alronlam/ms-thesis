from twitter_data.database import DBManager

def retrieve_all_tweet_objects_from_db(tweet_ids, verbose=False):
    tweet_objects = []
    for index, tweet_id in enumerate(tweet_ids):
        tweet_object = DBManager.get_or_add_tweet(tweet_id)
        if tweet_object:
            tweet_objects.append(tweet_object)

        if verbose:
            print("Retrieved {} ({} successful) / {} tweets from the database.".format(index+1, len(tweet_objects), len(tweet_ids)))

    return tweet_objects