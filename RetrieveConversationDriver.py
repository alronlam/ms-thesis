from community_detection import Utils

# json_tweet_ids = Utils.load_non_rt_tweet_ids_from_json_files("D:/DLSU/Masters/MS Thesis/data-2016/test")
from twitter_data.database import DBUtils

# senti_tweet_objects = Utils.load_tweet_objects_from_senti_csv_files('D:/DLSU/Masters/MS Thesis/data-2016/test')
# senti_no_rt_tweet_objects = Utils.load_non_rt_tweet_objects_from_senti_csv_files('D:/DLSU/Masters/MS Thesis/data-2016/test')
# print(len(senti_tweet_objects))
# print(len(senti_no_rt_tweet_objects))

def count_tweets_with_context(tweets):
    count = 0
    candidates = []
    for index, tweet in enumerate(tweets):
        if tweet.in_reply_to_status_id is not None:
            candidates.append(tweet.in_reply_to_status_id)
        print("Counting context: Looking for candidates {}/{}".format(index, len(tweets)))

    return len(DBUtils.retrieve_all_tweet_objects_from_db(candidates, verbose=True))


# brexit_tweets = Utils.load_tweet_objects_from_json_files("D:/DLSU/Masters/MS Thesis/data-2016/test")
# print(len(brexit_tweets))
# print(count_tweets_with_context(brexit_tweets))
# print(len([1 for tweet in brexit_tweets if len(DBUtils.retrieve_full_conversation(tweet.in_reply_to_status_id, [])) > 0]))

vanzo_tweet_ids = Utils.load_tweet_ids_from_vanzo_dataset()
print(len(vanzo_tweet_ids))
vanzo_tweets = DBUtils.retrieve_all_tweet_objects_from_db(vanzo_tweet_ids, verbose=True)
print(count_tweets_with_context(vanzo_tweets))