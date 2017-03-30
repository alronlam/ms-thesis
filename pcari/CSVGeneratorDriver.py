import csv
import os

from sentiment_analysis.preprocessing.PreProcessing import SplitWordByWhitespace, WordToLowercase, ReplaceURL, \
    ConcatWordArray, RemovePunctuationFromWords, ReplaceUsernameMention, RemoveRT, RemoveLetterRepetitions, \
    preprocess_tweet, RemoveTerm
from twitter_data.parsing.folders import FolderIO


categories = ["victim_identification_assistance", "raising_funds", "accounting_damage", "expressing_appreciation", "celebrification"]

# for category in categories:
category = "relevant_irrelevant"
data_source_dir ="C:/Users/user/PycharmProjects/ms-thesis/pcari/data/relevant_irrelevant"

preprocessors = [SplitWordByWhitespace(),
                 WordToLowercase(),
                 ReplaceURL(),
                 # ConcatWordArray(),
                 RemovePunctuationFromWords(),
                 ReplaceUsernameMention(),
                 RemoveRT(),
                 RemoveLetterRepetitions(),
                 RemoveTerm("yolanda"),
                 RemoveTerm("haiyan"),
                 RemoveTerm("victims"),
                 RemoveTerm("typhoon"),
                 ConcatWordArray()]

txt_files = FolderIO.get_files(data_source_dir, False, ".txt")

dataset_arr = []

for txt_file in txt_files:
    label = os.path.splitext(txt_file.name)[0]
    if label == "irrelevant":
        label="irrelevant"
    else:
        label="relevant"
    # if category not in label:
    #     label = "others"
    # else:
    #     label = category

    with txt_file.open(encoding="utf8") as input_file:
        tweet_texts = [line.rstrip() for line in input_file.readlines() if line.rstrip() is not None]
        dataset_arr.extend([(preprocess_tweet(tweet_text, preprocessors), label)for tweet_text in tweet_texts])


with open("data/{}.csv".format(category), "w", encoding="utf8", newline='') as output_file:
    csv_writer = csv.writer(output_file)
    for tuple in dataset_arr:
        csv_writer.writerow(tuple)
