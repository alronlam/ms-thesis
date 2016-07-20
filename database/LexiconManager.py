from database import DBManager

def get_sentiment_score(word):
    eng_score = DBManager.lexicon_so_collection.find_one({"eng":word})
    fil_score = DBManager.lexicon_so_collection.find_one({"fil":word})

    if fil_score:
        return 1 if fil_score['polarity'] == 'positive' else 0

    if eng_score:
        return 1 if eng_score['polarity'] == 'positive' else 0

    return 0 #neutral if word is not found

