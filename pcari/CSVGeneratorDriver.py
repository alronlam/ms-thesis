import csv
import os

from twitter_data.parsing.folders import FolderIO

data_source_dir ="C:/Users/user/PycharmProjects/ms-thesis/pcari/data/orig_data"

txt_files = FolderIO.get_files(data_source_dir, False, ".txt")

dataset_arr = []

for txt_file in txt_files:
    label = os.path.splitext(txt_file.name)[0]
    if "funds" not in label:
        label = "others"

    with txt_file.open(encoding="utf8") as input_file:
        tweet_texts = [line.rstrip() for line in input_file.readlines() if line.rstrip() is not None]
        dataset_arr.extend([(tweet_text, label)for tweet_text in tweet_texts])


with open("data/yolanda_nov2013_feb2014_dataset_funds_others.csv", "w", encoding="utf8", newline='') as output_file:
    csv_writer = csv.writer(output_file)
    for tuple in dataset_arr:
        csv_writer.writerow(tuple)
