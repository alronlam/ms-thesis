import pickle

def save_classifier_to_pickle(pickle_file_name, classifier):
    f = open(pickle_file_name, 'wb+')
    pickle.dump(classifier, f)
    f.close()

def load_classifier_from_pickle(pickle_file_name):
    try:
        f = open(pickle_file_name, 'rb')
        classifier = pickle.load(f)
        f.close()
        return classifier
    except:
        return None

