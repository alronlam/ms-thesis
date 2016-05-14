class ClassifiedTweet:

    def __init__(self, tweet_id, classification):
        self.tweet_id = tweet_id
        self.classification = classification

    def __str__(self):
        return "{}-{}".format(self.tweet_id, self.classification)

    def __repr__(self):
        return self.__str__()