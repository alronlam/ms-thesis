import string

from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from analysis.topic_modelling.TopicModeller import TopicModeller


class LDATopicModeller(TopicModeller):

    def __init__(self, num_topics=10, num_words=10):
        self.num_topics = num_topics
        self.num_words = num_words

        self.stop_words = set(stopwords.words('english'))
        self.excluded_chars = set(string.punctuation)
        self.lemma = WordNetLemmatizer()
        self.LDA = LdaModel

    def model_topics(self, document_list):
        document_list_cleaned = [self.preprocess_document(document) for document in document_list]
        dictionary = corpora.Dictionary(document_list_cleaned)
        doc_term_matrix = [dictionary.doc2bow(document) for document in document_list_cleaned]
        LDA_model = self.LDA(doc_term_matrix, num_topics=self.num_topics, id2word = dictionary, passes=50)
        return LDA_model.print_topics(num_topics=self.num_topics, num_words=self.num_words)


    def preprocess_document(self, document):
        stop_free = " ".join([i for i in document.lower().split() if i not in self.stop_words])
        punc_free = ''.join(ch for ch in stop_free if ch not in self.excluded_chars)
        normalized = " ".join(self.lemma.lemmatize(word) for word in punc_free.split())
        return normalized


