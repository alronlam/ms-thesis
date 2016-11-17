# CNN for the IMDB problem
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


from twitter_data.parsing.folders import FolderIO
from twitter_data.database import DBManager
from sentiment_analysis.evaluation import TSVParser
from word_embeddings import GoogleWordEmbedder
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
test_split = 0.33

#load dataset
tsv_files = FolderIO.get_files('D:/DLSU/Masters/MS Thesis/data-2016/Context-Based_Tweets/test', True, '.tsv')
conversations = TSVParser.parse_files_into_conversation_generator(tsv_files)

X_train = []
X_test = []
Y_train = []
Y_test = []

for index, conversation in enumerate(conversations):
    target_tweet = conversation[-1]
    print("{} - {}".format(index, target_tweet["tweet_id"]))
    tweet_object = DBManager.get_or_add_tweet(target_tweet["tweet_id"])
    if tweet_object:

        if target_tweet["class"] == 'negative':
            y = 0
        elif target_tweet["class"] == 'neutral':
            y = 1
        elif target_tweet["class"] == 'positive':
            y = 2

        embedded_word = GoogleWordEmbedder.google_embedding(tweet_object.text)
        X_train.append(embedded_word)
        X_test.append(embedded_word)

        Y_train.append(y)
        Y_test.append(embedded_word)
    if index > 50:
        break



# create the model
model = Sequential()
# model.add(Embedding(top_words, 32, input_length=max_words))
# model.add(Flatten())
model.add(Dense(300, activation='relu', input_dim=300))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=2, batch_size=128, verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))