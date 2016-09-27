from database import DBManager

def get_sentiment_score(word):
    eng_score = DBManager.lexicon_so_collection.find_one({"eng":word})
    fil_score = DBManager.lexicon_so_collection.find_one({"fil":word})

    if fil_score:
        if fil_score['polarity'] == 'positive':
            return 1
        elif fil_score['polarity'] == 'negative':
            return -1
        else:
            return 0

    if eng_score:
        if eng_score['polarity'] == 'positive':
            return 1
        elif eng_score['polarity'] == 'negative':
            return -1
        else:
            return 0

    return 0 #neutral if word is not found

