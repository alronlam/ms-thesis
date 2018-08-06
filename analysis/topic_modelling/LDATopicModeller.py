import string

import numpy
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from analysis.topic_modelling.TopicModeller import TopicModeller


class LDATopicModeller(TopicModeller):

    def __init__(self, num_topics=5, num_words=10):
        self.num_topics = num_topics
        self.num_words = num_words

        self.stop_words = set(stopwords.words('english')).union(stopwords.words('french')).union(
            stopwords.words('spanish'))
        self.excluded_chars = set(string.punctuation)
        self.lemma = WordNetLemmatizer()
        self.LDA = LdaModel

        self.lda_model = None
        self.dictionary = None

    def generate_topic_model_string(self, document_list):
        self.generate_topic_models(document_list)
        return self.generate_topic_model_string_from_tuples()

    def generate_topic_model_string_from_tuples(self):
        topic_string = []
        for index, topic_probability_list in self.word_weight_tuples:

            topics = [topic for topic, probability in topic_probability_list]
            concatenated_topics = " ".join(topics)
            topic_string.append("Topic {} - {}".format(index, concatenated_topics))
            for topic, probability in topic_probability_list:
                topic_string.append("{} - {}".format(topic, probability))
            topic_string.append("\n")

        return "\n".join(topic_string)

    def generate_topic_models(self, document_list, verbose=False):
        document_list_cleaned = [self.preprocess_document(document) for document in document_list]
        dictionary = corpora.Dictionary(document_list_cleaned)
        doc_term_matrix = [dictionary.doc2bow(document) for document in document_list_cleaned]
        lda_model = self.LDA(doc_term_matrix, num_topics=self.num_topics, id2word=dictionary, passes=50, iterations=100,
                             eval_every=1)
        word_weight_tuples = lda_model.show_topics(num_topics=self.num_topics, num_words=self.num_words, log=False,
                                                   formatted=False)

        self.lda_model = lda_model
        self.dictionary = dictionary
        self.word_weight_tuples = word_weight_tuples

    def closest_topic(self, document, with_score=False):
        document = self.preprocess_document(document)
        bow = self.dictionary.doc2bow(document)
        topic_probabilities = self.lda_model.get_document_topics(bow)

        topics = [topic for (topic, probability) in topic_probabilities]
        probabilities = numpy.array([probability for (topic, probability) in topic_probabilities])
        max_index = probabilities.argmax()

        if with_score:
            return topics[max_index], probabilities[max_index]
        else:
            return topics[max_index]

    def preprocess_document(self, document):
        stop_free = " ".join([i for i in document.lower().split() if i not in self.stop_words])
        punc_free = ''.join(ch for ch in stop_free if ch not in self.excluded_chars)
        normalized = " ".join(self.lemma.lemmatize(word) for word in punc_free.split())
        return normalized.split()
