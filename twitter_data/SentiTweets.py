import re


class SentiTweetAdapter(object):
    def __init__(self, text, user_screen_name):
        self.text = text
        self.user = SentiTweetAdapterUser()
        self.user.screen_name = user_screen_name
        self.user.id_str = user_screen_name

        # initialize entities here

        # mentions
        mentions = get_mentions_from_tweet_text(text)
        mentions_dict_list = [{"id_str":mention, "screen_name":mention} for mention in mentions]

        # hashtags
        hashtags = get_hashtags_from_tweet_text(text)
        hashtag_dict_list = [{"text":hashtag} for hashtag in hashtags]

        self.entities = {'user_mentions': mentions_dict_list, 'hashtags': hashtag_dict_list}

    def __str__(self):
        return " - ".join([self.user_screen_name, self.text])

    def __repr__(self):
        return " - ".join([self.user_screen_name, self.text])

class SentiTweetAdapterUser(object):
    pass


#TODO double check if this has to have # or not
def get_hashtags_from_tweet_text(text):
    words = text.split()
    hashtags = [word for word in words if re.match("#\w+", word)]
    return hashtags

#TODO double check if this has to have @ or not
def get_mentions_from_tweet_text(text):
    words = text.split()
    mentions = [word for word in words if re.match("@\w+", word)]
    return mentions