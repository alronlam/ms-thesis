class ReplyNode:

    def __init__(self, tweet_id):
        self.tweet_id = tweet_id
        self.children = []

    def add_child(self, tweet_id):
        self.children.append(tweet_id)