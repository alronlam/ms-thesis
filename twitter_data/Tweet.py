import re


class SentiTweetAdapter(object):
    def __init__(self, text, user_screen_name):
        self.text = text
        self.user = SentiTweetAdapterUser()
        self.user.screen_name = user_screen_name
        self.user.id_str = user_screen_name

        # initialize entities here
        mentions = get_mentions_from_tweet_text(text)
        mentions_dict_list = [{"id_str":mention, "screen_name":mention} for mention in mentions]
        self.entities = {'user_mentions': mentions_dict_list}

    def __str__(self):
        return " - ".join([self.user_screen_name, self.text])

    def __repr__(self):
        return " - ".join([self.user_screen_name, self.text])

class SentiTweetAdapterUser(object):
    pass

def get_mentions_from_tweet_text(text):
    words = text.split()
    mentions = [word for word in words if re.match("@\w+", word)]
    return mentions
    #TODO double check if this has to have @ or not
    return [{"id_str":1, "screen_name": "sample2"},{"id_str":2, "screen_name":"sample2"}]