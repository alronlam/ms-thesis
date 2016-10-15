from datetime import datetime

import numpy
from sklearn import metrics

from twitter_data.parsing.csv_parser import CSVParser
from twitter_data.parsing.folders import FolderIO
from sentiment_analysis import SentimentClassifier

def write_metrics_file(actual_arr, predicted_arr, metrics_file_name):
    metrics_file = open(metrics_file_name, 'w')
    metrics_file.write('Total: {}\n'.format(actual_arr.__len__()))
    metrics_file.write('Accuracy: {}\n'.format(metrics.accuracy_score(actual_arr, predicted_arr)))
    metrics_file.write(metrics.classification_report(actual_arr, predicted_arr))
    metrics_file.write('\n')
    metrics_file.write(numpy.array_str(metrics.confusion_matrix(actual_arr, predicted_arr))) # ordering is alphabetical order of label names
    metrics_file.write('\n')
    metrics_file.close()

def test_senti_election_data():
    csv_files = FolderIO.get_files('D:/DLSU/Masters/MS Thesis/data-2016/senti_election_data/csv_files/test', False, '.csv')

    csv_rows = CSVParser.parse_files_into_csv_row_generator(csv_files, True)

    classifier = SentimentClassifier.ANEWLexiconClassifier()

    correct = 0
    total = 0

    actual_arr = []
    predicted_arr = []

    metrics_file_name = 'metrics-{}.txt'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    for csv_row in csv_rows:
        try:
            actual_class = csv_row[6]
            predicted_class = classifier.classify_sentiment(csv_row[1])

            if actual_class.lower() == predicted_class.lower():
                correct += 1
            total += 1

            print('{:.2f} = {}/{}'.format(correct/total, correct, total))

            actual_arr.append(actual_class.lower())
            predicted_arr.append(predicted_class.lower())

            if total % 500 == 0:
                write_metrics_file(actual_arr, predicted_arr, metrics_file_name)

        except Exception as e:
            print(e)
            pass


# test_senti_election_data()

