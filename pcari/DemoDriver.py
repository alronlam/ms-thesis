import pickle

from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences

from sentiment_analysis.preprocessing import PreProcessing
from sentiment_analysis.preprocessing.PreProcessing import *

MAX_SEQUENCE_LENGTH = 32


relevance_model_path = "relevant_irrelevant_model_fold_0.json"
relevance_weights_path = "relevant_irrelevant_weights_fold_0.h5"
relevance_tokenizer_path = "data/relevant_irrelevant.npz-tokenizer.pickle"
relevance_categories = ["irrelevant", "relevant"]

categories_model_path = "5_categories_model_fold_0.json"
categories_weights_path = "5_categories_weights_fold_0.h5"
categories_tokenizer_path = "data/relevant_irrelevant.npz-tokenizer.pickle"
five_categories = ["victim identification and assistance",
                   "raising funds",
                   "accounting damage",
                   "expressing appreciation",
                   "celebrification"]


def load_model_and_tokenizer(model_path, weights_path, tokenizer_path):
    model = model_from_json(open(model_path, "r").readlines()[0])
    model.load_weights(weights_path)
    tokenizer = pickle.load(open(tokenizer_path,"rb"))

    return (model, tokenizer)


(relevance_model, relevance_tokenizer) = load_model_and_tokenizer(relevance_model_path, relevance_weights_path, relevance_tokenizer_path)
(categories_model, categories_tokenizer) = load_model_and_tokenizer(categories_model_path, categories_weights_path, categories_tokenizer_path)

preprocessors = [SplitWordByWhitespace(),
                 WordToLowercase(),
                 ReplaceURL(),
                 RemovePunctuationFromWords(),
                 ReplaceUsernameMention(),
                 RemoveRT(),
                 RemoveLetterRepetitions(),
                 ConcatWordArray()]

def classify(raw_input, model, tokenizer, categories):
    preprocessed_input = PreProcessing.preprocess_tweet(raw_input, preprocessors)
    tokenized_input = tokenizer.texts_to_sequences([preprocessed_input])
    tokenized_input = pad_sequences(tokenized_input, maxlen=MAX_SEQUENCE_LENGTH)

    predicted_probabilities = model.predict([tokenized_input], batch_size=1, verbose=False)[0].tolist()
    max_index = predicted_probabilities.index(max(predicted_probabilities))
    category = categories[max_index]

    return category


while(True):
    raw_input = input("Enter text: ")

    if classify(raw_input, relevance_model, relevance_tokenizer, relevance_categories) == "relevant":
        print("relevant: {}".format(classify(raw_input, categories_model, categories_tokenizer, five_categories)))
    else:
        print("irrelevant")