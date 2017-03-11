import string

import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from analysis.topic_modelling.TopicModeller import TopicModeller


class LDATopicModeller(TopicModeller):

    def __init__(self, num_topics=5, num_words=20):
        self.num_topics = num_topics
        self.num_words = num_words

        self.stop_words = set(stopwords.words('english'))
        self.excluded_chars = set(string.punctuation)
        self.lemma = WordNetLemmatizer()
        self.LDA = LdaModel

    def model_topics(self, document_list):
        try:
            document_list_cleaned = [self.preprocess_document(document) for document in document_list]
            dictionary = corpora.Dictionary(document_list_cleaned)
            doc_term_matrix = [dictionary.doc2bow(document) for document in document_list_cleaned]
            LDA_model = self.LDA(doc_term_matrix, num_topics=self.num_topics, id2word = dictionary, passes=50)
            topic_probability_tuples =  LDA_model.show_topics(num_topics=self.num_topics, num_words=self.num_words, log=False, formatted=False)

            topic_string = []
            for index, topic_probability_list in topic_probability_tuples:
                topic_string.append("Topic {}".format(index))
                for topic, probability in topic_probability_list:
                    topic_string.append("{} - {}".format(topic, probability))
                topic_string.append("\n")
            return "\n".join(topic_string)
        except Exception as e:
            return None


    def preprocess_document(self, document):
        stop_free = " ".join([i for i in document.lower().split() if i not in self.stop_words])
        punc_free = ''.join(ch for ch in stop_free if ch not in self.excluded_chars)
        normalized = " ".join(self.lemma.lemmatize(word) for word in punc_free.split())
        return normalized.split()


