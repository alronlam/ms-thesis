import abc


class TopicModeller(object):

    @abc.abstractmethod
    def generate_topic_model_string(self, document_list):
        """
        :param document_list: list of text documents for topic modelling
        :return:
        """

    @abc.abstractmethod
    def generate_topic_models(self, document_list):
        """
        :param document_list: list of text documents for topic modelling
        :return:
        """