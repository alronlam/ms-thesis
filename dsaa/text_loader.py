import csv

from twitter_data.parsing.folders import FolderIO


def load_community_docs(dir):
    csv_files = FolderIO.get_files(dir, False, '.csv')
    community_docs = []
    for csv_file in sorted(csv_files):
        print(csv_file)
        with csv_file.open(encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter="\n")
            community_docs.append([row[0] for row in csv_reader if len(row) > 0])

    return community_docs