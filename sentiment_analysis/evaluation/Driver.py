from sentiment_analysis.evaluation import TSVParser
from sentiment_analysis import SentimentClassifier
from twitter_data.parsing.folders import FolderIO
from twitter_data.database import DBManager
from sentiment_analysis.subjectivity.SubjectivityClassifier import MLSubjectivityClassifier
import pickle
from datetime import datetime

import numpy
from sklearn import metrics

def write_metrics_file(actual_arr, predicted_arr, metrics_file_name):
    metrics_file = open(metrics_file_name, 'w')
    metrics_file.write('Total: {}\n'.format(actual_arr.__len__()))
    metrics_file.write('Accuracy: {}\n'.format(metrics.accuracy_score(actual_arr, predicted_arr)))
    try:
        metrics_file.write(metrics.classification_report(actual_arr, predicted_arr))
    except Exception as e:
        print(e)
        pass
    metrics_file.write('\n')
    metrics_file.write(numpy.array_str(metrics.confusion_matrix(actual_arr, predicted_arr))) # ordering is alphabetical order of label names
    metrics_file.write('\n')
    metrics_file.close()


def count_vanzo_eng_dataset():
    tsv_files = FolderIO.get_files('D:/DLSU/Masters/MS Thesis/data-2016/Context-Based_Tweets/test', True, '.tsv')
    conversations = TSVParser.parse_files_into_conversation_generator(tsv_files)
    print([x for x in conversations].__len__())

def test_vanzo_eng_dataset(classifier, subjectivity_classifier):

    tsv_files = FolderIO.get_files('D:/DLSU/Masters/MS Thesis/data-2016/Context-Based_Tweets/conv_test', True, '.tsv')
    conversations = TSVParser.parse_files_into_conversation_generator(tsv_files)

    metrics_file_name = 'metrics-vanzo-eng-{}-{}-{}.txt'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), classifier.get_name(), "w_subj" if subjectivity_classifier else "" )


    actual_arr = []
    predicted_arr = []

    for index, conversation in enumerate(conversations):
        target_tweet = conversation[-1]

        print("{} - {}".format(index, target_tweet["tweet_id"]))
        tweet_object = DBManager.get_or_add_tweet(target_tweet["tweet_id"])
        if tweet_object:

            if subjectivity_classifier and subjectivity_classifier.classify_subjectivity(tweet_object.text) == 'objective':
                predicted_class = 'neutral'
            else:
                predicted_class = classifier.classify_sentiment(tweet_object.text, {'conversation': conversation})

            actual_class = target_tweet["class"]

            predicted_arr.append(predicted_class)
            actual_arr.append(actual_class)

            # print("{} vs {}".format( actual_class,predicted_class))

        if index % 100 == 0 and index > 0:
            pickle.dump((actual_arr, predicted_arr), open( "{}.pickle".format(metrics_file_name), "wb" ) )
            write_metrics_file(actual_arr, predicted_arr, metrics_file_name)

    pickle.dump((actual_arr, predicted_arr), open( "{}.pickle".format(metrics_file_name), "wb" ) )
    write_metrics_file(actual_arr, predicted_arr, metrics_file_name)

wiebe_lexicon_classifier = SentimentClassifier.WiebeLexiconClassifier()
anew_lexicon_classifier = SentimentClassifier.ANEWLexiconClassifier()
afinn_classifier = SentimentClassifier.AFINNLexiconClassifier()
globe_ml_classifier = SentimentClassifier.MLClassifier("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/unigram_feature_extractor.pickle", "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/nb_classifier.pickle.pickle")
subjectivity_classifier = MLSubjectivityClassifier('C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/subjectivity/subj_unigram_feature_extractor_vanzo_conv_train.pickle', 'C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/subjectivity/subj_nb_classifier_vanzo_conv_train.pickle' )
nb_unigram_classifier = SentimentClassifier.MLClassifier("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/sa_nb_unigram_feature_extractor_vanzo_conv_train.pickle", "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/sa_nb_classifier_vanzo_conv_train.pickle")
svm_unigram_classifier = SentimentClassifier.MLClassifier("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/sa_svm_unigram_feature_extractor_vanzo_conv_train.pickle", "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/sa_svm_classifier_vanzo_conv_train.pickle")

# test_vanzo_eng_dataset(afinn_classifier, None)
# test_vanzo_eng_dataset(anew_lexicon_classifier, None)
# test_vanzo_eng_dataset(wiebe_lexicon_classifier, None)
# test_vanzo_eng_dataset(globe_ml_classifier, None)
# test_vanzo_eng_dataset(nb_unigram_classifier, None)
# test_vanzo_eng_dataset(svm_unigram_classifier, None)


# test_vanzo_eng_dataset(afinn_classifier, subjectivity_classifier)
# test_vanzo_eng_dataset(anew_lexicon_classifier, subjectivity_classifier)
# test_vanzo_eng_dataset(wiebe_lexicon_classifier, subjectivity_classifier)
# test_vanzo_eng_dataset(globe_ml_classifier, subjectivity_classifier)
# test_vanzo_eng_dataset(nb_unigram_classifier, subjectivity_classifier)
# test_vanzo_eng_dataset(svm_unigram_classifier, subjectivity_classifier)

# corpus_pickle_file_name = 'C:/Users/user/PycharmProjects/ms-thesis/word_embeddings/vanzo_corpus_w2v.pickle'
# corpus_bin_file_name = 'D:/DLSU/Masters/MS Thesis/Resources/GoogleNews-vectors-negative300.bin'
# classifier_pickle_file_name = 'C:/Users/user/PycharmProjects/ms-thesis/word_embeddings/svm_classifier.pickle'
# scaler_pickle_file_name = 'C:/Users/user/PycharmProjects/ms-thesis/word_embeddings/svm_scaler.pickle'

# print("Initializing classifier")
# conversational_context_clasifier = SentimentClassifier.ConversationalContextClassifier(corpus_bin_file_name, classifier_pickle_file_name, scaler_pickle_file_name)
# print("Finished loading classifier")
# test_vanzo_eng_dataset(conversational_context_clasifier, subjectivity_classifier)

keras_tokenizer_pickle_path = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/feature_extraction/word_embeddings/tokenizer-vanzo_word_sequence_concat_glove_200d.npz.pickle"
keras_classifier_json_path = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/neural_nets/keras_model_no_context.json"
keras_classifier_weights_path = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/neural_nets/keras_model_no_context_weights.h5"
keras_classifier = SentimentClassifier.KerasClassifier(keras_tokenizer_pickle_path, keras_classifier_json_path, keras_classifier_weights_path)


def test_classify(classifier):
    while True:
        print("Enter text:")
        text = input()
        print(classifier.classify_sentiment(text, {}))

test_classify(keras_classifier)