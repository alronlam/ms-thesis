import pickle
import numpy
from sklearn.preprocessing import StandardScaler


def train_sa():

    train_data = numpy.load("C:/Users/user/PycharmProjects/ms-thesis/word_embeddings/vanzo_train.npz")
    test_data = numpy.load("C:/Users/user/PycharmProjects/ms-thesis/word_embeddings/vanzo_test.npz")

    X_train = train_data["X"]
    Y_train = train_data["Y"]

    X_test = test_data["X"]
    Y_test = test_data["Y"]

    def convert_numeric_to_string(y):
        if y == 0:
            return "negative"
        if y == 1:
            return "neutral"
        if y == 2:
            return "positive"

    # Change 0, 1, 2 to negative, neutral, positive
    Y_train = [convert_numeric_to_string(y) for y in Y_train]
    Y_test = [convert_numeric_to_string(y) for y in Y_test]

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.linear_model import SGDClassifier
    import sklearn
    from sklearn import naive_bayes

    # lr = naive_bayes.GaussianNB()
    lr = sklearn.svm.SVC()
    # lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(X_train, Y_train)

    pickle.dump(scaler, open('svm_scaler.pickle', "wb"))
    pickle.dump(lr, open("svm_classifier.pickle", "wb"))

    print('Test Accuracy: {}'.format(lr.score(X_test, Y_test)))

# train_sa()

def train_subjectivity():

    train_data = numpy.load("C:/Users/user/PycharmProjects/ms-thesis/word_embeddings/vanzo_train.npz")
    test_data = numpy.load("C:/Users/user/PycharmProjects/ms-thesis/word_embeddings/vanzo_test.npz")

    X_train = train_data["X"]
    Y_train = train_data["Y"]

    X_test = test_data["X"]
    Y_test = test_data["Y"]

    def convert_numeric_to_string(y):
        if y == 1:
            return "objective"
        else:
            return "subjective"

    # Change 0, 1, 2 to negative, neutral, positive
    Y_train = [convert_numeric_to_string(y) for y in Y_train]
    Y_test = [convert_numeric_to_string(y) for y in Y_test]

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.linear_model import SGDClassifier
    import sklearn
    from sklearn import naive_bayes

    # lr = naive_bayes.GaussianNB()
    lr = sklearn.svm.SVC()
    # lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(X_train, Y_train)

    pickle.dump(scaler, open('subj_svm_scaler.pickle', "wb"))
    pickle.dump(lr, open("subj_svm_classifier.pickle", "wb"))

    print('Test Accuracy: {}'.format(lr.score(X_test, Y_test)))


train_subjectivity()