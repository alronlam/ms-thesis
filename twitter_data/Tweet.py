class SentiTweetAdapter(object):
    def __init__(self, text, user_screen_name):
        self.text = text
        self.user_screen_name = user_screen_name

        # initialize entities here
        # self.entities = {'mentions': get_mentions_from_tweet_text()}

    def __str__(self):
        return " - ".join([self.user_screen_name, self.text])

    def __repr__(self):
        return " - ".join([self.user_screen_name, self.text])

def get_mentions_from_tweet_text(text):
    #TODO double check if this has to have @ or not
    return [{"id_str":1, "screen_name": "sample2"},{"id_str":2, "screen_name":"sample2"}]