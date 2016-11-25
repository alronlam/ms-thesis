import pickle
import numpy
from sklearn.preprocessing import StandardScaler

train_data = numpy.load("C:/Users/user/PycharmProjects/ms-thesis/word_embeddings/vanzo_train.npz")
test_data = numpy.load("C:/Users/user/PycharmProjects/ms-thesis/word_embeddings/vanzo_test.npz")

X_train = train_data["X"]
Y_train = train_data["Y"]

X_test = test_data["X"]
Y_test = test_data["Y"]

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

# Create ROC curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# pred_probas = lr.predict_proba(X_test)[:, 1]
#
# fpr, tpr, _ = roc_curve(Y_test, pred_probas)
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.legend(loc='lower right')
#
# plt.show()