import abc

class SubjectivityClassifier(object):

    @abc.abstractmethod
    def classify_subjectivity(self, text):
        """
        :param text: string to be analyzed
        :return: "negative" "positive" or "neutral"
        """


# class MLSubjectivityClassifier(SubjectivityClassifier):
