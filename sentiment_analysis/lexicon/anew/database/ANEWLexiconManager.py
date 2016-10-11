from twitter_data.database import DBManager

def get_sentiment_score(word):
    eng_score = DBManager.anew_lexicon_collection.find_one({"english": word})
    fil_score = DBManager.anew_lexicon_collection.find_one({"filipino":{"$in": [word]}})

    if eng_score:
        return eng_score["valence"] - 5
    if fil_score:
        return fil_score["valence"] - 5

    return 0 #neutral if word is not found