from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
from twitter_data.parsing.folders import FolderIO
from sentiment_analysis.evaluation import TSVParser
from twitter_data.database import DBManager


# def write_pos_tweets_to_txt_file():
#     pos_tweets_file = open('pos_tweets.txt', 'w')
#     neg_tweets_file = open('neg_tweets.txt', 'w')
#
#     tsv_files = FolderIO.get_files('D:/DLSU/Masters/MS Thesis/data-2016/Context-Based_Tweets/test', True, '.tsv')
#     conversations = TSVParser.parse_files_into_conversation_generator(tsv_files)
#
#     for index, conversation in enumerate(conversations):
#         target_tweet = conversation[-1]
#         tweet_object = DBManager.get_or_add_tweet(target_tweet["tweet_id"])
#         if tweet_object:
#             if target_tweet["class"] == 'positive':
#                 pos_tweets_file.write("{}\n".format(tweet_object.text))
#             elif target_tweet["class"] == 'negative':
#                 neg_tweets_file.write("{}\n".format(tweet_object.text))
#             print(index)
#
#     pos_tweets_file.close()
#     neg_tweets_file.close()

# write_pos_tweets_to_txt_file()
#

from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec

print("READING TWEETS")
with open('pos_tweets.txt', 'r') as infile:
    pos_tweets = infile.readlines()

with open('neg_tweets.txt', 'r') as infile:
    neg_tweets = infile.readlines()

print("CONCATENATING")
#use 1 for positive sentiment, 0 for negative
y = np.concatenate((np.ones(len(pos_tweets)), np.zeros(len(neg_tweets))))

print("PARTITIONING DATASET")
x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_tweets, neg_tweets)), y, test_size=0.2)

#Do some very minor text preprocessing
from sentiment_analysis.preprocessing import PreProcessing
def cleanText(corpus):
    preprocessors = [PreProcessing.SplitWordByWhitespace(), PreProcessing.WordToLowercase(),
                              PreProcessing.RemovePunctuationFromWords()]

    clean_text = []
    for z in corpus:
        for preprocessor in preprocessors:
            z = preprocessor.preprocess_tweet(z)
        clean_text.append(z)
    # corpus = [z.lower().replace('\n','').split() for z in corpus]
    return clean_text

print("GOING TO CLEAN TEXT")
x_train = cleanText(x_train)
x_test = cleanText(x_test)

print("BUILDING VOCAB")
n_dim = 300
#Initialize model and build vocab
vanzo_w2v = Word2Vec(size=n_dim, min_count=10)
vanzo_w2v.build_vocab(x_train)

print("TRAINING")
#Train the model over train_reviews (this may take several minutes)
vanzo_w2v.train(x_train)

import pickle
pickle.dump(vanzo_w2v, open( "vanzo_corpus_w2v.pickle", "wb"))

def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += vanzo_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

from sklearn.preprocessing import scale
train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
train_vecs = scale(train_vecs)

#Train word2vec on test tweets
vanzo_w2v.train(x_test)

test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
test_vecs = scale(test_vecs)

from sklearn.linear_model import SGDClassifier
from sentiment_analysis.subjectivity import SubjectivityClassifier

lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)


pickle.dump(lr, open( "sgd_classifier.pickle", "wb"))

print('Test Accuracy: {}'.format(lr.score(test_vecs, y_test)))

#Create ROC curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

pred_probas = lr.predict_proba(test_vecs)[:,1]

fpr,tpr,_ = roc_curve(y_test, pred_probas)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')

plt.show()
