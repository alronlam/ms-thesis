from sentiment_analysis.preprocessing import PreProcessing
from gensim.models.word2vec import Word2Vec
import numpy as np

w2v_model = Word2Vec.load_word2vec_format('D:/DLSU/Masters/MS Thesis/Resources/GoogleNews-vectors-negative300.bin', binary=True)
SIZE = 300
def clean_text(text):
    preprocessors = [PreProcessing.SplitWordByWhitespace(), PreProcessing.WordToLowercase(),
                     PreProcessing.RemovePunctuationFromWords()]

    for preprocessor in preprocessors:
        text = preprocessor.preprocess_tweet(text)

    return text

def google_embedding(text):
    text = clean_text(text)
    vec = np.zeros(SIZE).reshape((1, SIZE))
    count = 0.
    for word in text:
        try:
            vec += w2v_model[word].reshape((1, SIZE))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec
