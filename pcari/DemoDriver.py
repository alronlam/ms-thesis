import pickle

from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences

from sentiment_analysis.preprocessing import PreProcessing
from sentiment_analysis.preprocessing.PreProcessing import *


MAX_SEQUENCE_LENGTH = 32
json_string = open("relevant_irrelevant_model_fold_0.json", "r").readlines()[0]
weights_path = "relevant_irrelevant_weights_fold_0.h5"
tokenizer_path = "data/relevant_irrelevant.npz-tokenizer.pickle"


model = model_from_json(json_string)
model.load_weights(weights_path)
tokenizer = pickle.load(open(tokenizer_path,"rb"))

preprocessors = [SplitWordByWhitespace(),
                 WordToLowercase(),
                 ReplaceURL(),
                 RemovePunctuationFromWords(),
                 ReplaceUsernameMention(),
                 RemoveRT(),
                 RemoveLetterRepetitions(),
                 RemoveTerm("yolanda"),
                 RemoveTerm("haiyan"),
                 RemoveTerm("victims"),
                 RemoveTerm("typhoon"),
                 ConcatWordArray()]

def convert_num_to_category(num):
    if num == 0:
        return "irrelevant"
    return "relevant"
while(True):
    raw_input = input("Enter text: ")
    preprocessed_input = PreProcessing.preprocess_tweet(raw_input, preprocessors)
    tokenized_input = tokenizer.texts_to_sequences([preprocessed_input])
    tokenized_input = pad_sequences(tokenized_input, maxlen=MAX_SEQUENCE_LENGTH)

    predicted_probabilities = model.predict([tokenized_input], batch_size=1, verbose=False)[0].tolist()
    max_index = predicted_probabilities.index(max(predicted_probabilities))
    category = convert_num_to_category(max_index)

    print(category)