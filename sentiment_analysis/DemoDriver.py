from sentiment_analysis import SentimentClassifier
from sentiment_analysis.subjectivity.SubjectivityClassifier import MLSubjectivityClassifier

wiebe_lexicon_classifier = SentimentClassifier.WiebeLexiconClassifier()
anew_lexicon_classifier = SentimentClassifier.ANEWLexiconClassifier()
afinn_classifier = SentimentClassifier.AFINNLexiconClassifier()
globe_ml_classifier = SentimentClassifier.MLClassifier("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/unigram_feature_extractor.pickle", "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/nb_classifier.pickle.pickle")
subjectivity_classifier = MLSubjectivityClassifier('C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/subjectivity/subj_unigram_feature_extractor_vanzo_conv_train.pickle', 'C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/subjectivity/subj_nb_classifier_vanzo_conv_train.pickle' )
nb_unigram_classifier = SentimentClassifier.MLClassifier("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/sa_nb_unigram_feature_extractor_vanzo_conv_train.pickle", "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/sa_nb_classifier_vanzo_conv_train.pickle")
svm_unigram_classifier = SentimentClassifier.MLClassifier("C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/sa_svm_unigram_feature_extractor_vanzo_conv_train.pickle", "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/sa_svm_classifier_vanzo_conv_train.pickle")

keras_tokenizer_pickle_path = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/feature_extraction/word_embeddings/tokenizer-vanzo_word_sequence_concat_glove_200d.npz.pickle"
keras_classifier_json_path = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/neural_nets/keras_model_no_context.json"
keras_classifier_weights_path = "C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/machine_learning/neural_nets/keras_model_no_context_weights.h5"
keras_classifier = SentimentClassifier.KerasClassifier(keras_tokenizer_pickle_path, keras_classifier_json_path, keras_classifier_weights_path)



def test_classify(classifier):
    while True:
        print("Enter text:")
        text = input()
        print(classifier.classify_sentiment(text, {}))
        print()


test_classify(afinn_classifier)