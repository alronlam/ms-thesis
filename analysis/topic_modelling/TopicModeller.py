import abc


class TopicModeller(object):

    @abc.abstractmethod
    def model_topics(self, document_list):
        """
        :param document_list: list of text documents for topic modelling
        :return:
        """
