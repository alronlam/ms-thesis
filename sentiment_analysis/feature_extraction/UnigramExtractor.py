
class UnigramExtractor(FeatureExtractorBase):

    def extract_features(self, document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features
