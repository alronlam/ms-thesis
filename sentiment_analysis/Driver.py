import nltk
from csv_parser.CSVParser import CSVParser
from nltk.classify.naivebayes import NaiveBayesClassifier
from sentiment_analysis.feature_extraction.UnigramExtractor import UnigramExtractor

def train(labeled_featuresets, estimator=nltk.probability.ELEProbDist):
    label_freqdist = nltk.FreqDist([label for words, label in labeled_featuresets])
    label_probdist = estimator(label_freqdist)

    feature_probidst = {}

    return NaiveBayesClassifier(label_probdist, feature_probidst)


row_generator = CSVParser.parse_file_into_csv_row_generator('sa_training_data/globe_dataset.csv')

tweets = [([e.lower() for e in row[2].split() if len(e) >= 3], row[1]) for row in row_generator]

unigram_extractor = UnigramExtractor(tweets)

training_set = nltk.classify.apply_features(unigram_extractor.extract_features, tweets)
# print(training_set)
classifier = train(training_set)

tweet = "Larry is my friend"

print(classifier.classify(unigram_extractor.extract_features(tweet.split())))




def test_nltk():

    def get_words_in_tweets(tweets):
        all_words = []
        for (words, sentiment) in tweets:
            all_words.extend(words)
        return all_words

    def get_word_features(wordlist):
        wordlist = nltk.FreqDist(wordlist)
        word_features = wordlist.keys()
        return word_features

    def extract_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features

    def train(labeled_featuresets, estimator=nltk.probability.ELEProbDist):
        label_freqdist = nltk.FreqDist([label for words, label in labeled_featuresets])
        label_probdist = estimator(label_freqdist)

        feature_probidst = {}

        return NaiveBayesClassifier(label_probdist, feature_probidst)

    pos_tweets = [('I love this car', 'positive'),
                  ('This view is amazing', 'positive'),
                  ('I feel great this morning', 'positive'),
                  ('I am so excited about the concert', 'positive'),
                  ('He is my best friend', 'positive')]

    neg_tweets = [('I do not like this car', 'negative'),
                  ('This view is horrible', 'negative'),
                  ('I feel tired this morning', 'negative'),
                  ('I am not looking forward to the concert', 'negative'),
                  ('He is my enemy', 'negative')]


    tweets = []
    for (words, sentiment) in pos_tweets + neg_tweets:
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        tweets.append((words_filtered, sentiment))

    word_features = get_word_features(get_words_in_tweets(tweets))

    training_set = nltk.classify.apply_features(extract_features, tweets)
    # print(training_set)
    classifier = train(training_set)

    tweet = "Larry is my friend"

    print(classifier.classify(extract_features(tweet.split())))