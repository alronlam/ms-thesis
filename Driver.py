import datetime

from data_structures.Node import Node
from foldersio.FolderIO import FolderIO
from jsonparser.JSONParser import JSONParser
from tweets.TweetUtils import TweetUtils
from operator import itemgetter
import itertools

def count_nodes(node):
    count = 0
    if node is not None:
        count += 1
    for child in node.children:
        count += count_nodes(child)
    return count


def main():
    folderIO = FolderIO()
    files = folderIO.get_files("D:/DLSU/Masters/MS Thesis/data-2016/for_processing/", False, ".jsonparser")

    print("Found {} files.".format(len(files)))

    file_stats = open('results_stats.txt', 'a')
    file_summary = open('results_summary.txt', 'a')
    file_full = open('results_full.txt', 'a')
    file_frequency = open('results_frequency.txt', 'a')

    max_count = 0
    max_tweet_id = None
    max_node = None

    json_parser = JSONParser()
    for file in files:

        print("\nProcessing {}".format(file))

        # Append date-time to the result files
        file_stats.write('\n{}-{}\n'.format(datetime.datetime.now(), file.name))
        file_summary.write('\n{}-{}\n'.format(datetime.datetime.now(), file.name))
        file_full.write('\n{}-{}\n'.format(datetime.datetime.now(), file.name))
        file_frequency.write('\n{}-{}\n'.format(datetime.datetime.now(), file.name))

        thread_length_freq = {}
        processed_tweet_ids = set()
        tweet_helper = TweetUtils()
        api = tweet_helper.api
        lines_processed = 0
        tweets_processed = 0

        for tweet_json in json_parser.parse_file_into_json_generator(file):

            curr_tweet = tweet_helper.retrieve_tweet(tweet_json["id"])
            lines_processed += 1

            if curr_tweet is not None and curr_tweet.id not in processed_tweet_ids:

                processed_tweet_ids.add(curr_tweet.id)

                curr_reply_thread = tweet_helper.list_reply_ancestors(curr_tweet)
                curr_reply_thread_count = len(curr_reply_thread)

                thread_length_freq[curr_reply_thread_count] = thread_length_freq.get(curr_reply_thread_count, 0) + 1

                if curr_reply_thread_count > max_count:
                    max_count = curr_reply_thread_count
                    max_tweet_id = curr_tweet.id
                    file_summary.write("{}:\n{}\n\n".format(max_count, "\n".join(str(reply.id) for reply in curr_reply_thread)))
                    file_summary.flush()

                if curr_reply_thread_count >= 3:
                    file_full.write("{}:\n{}\n\n".format(curr_reply_thread_count, "\n".join(str(("@"+reply.user.screen_name + ": "+str(reply.text)).encode("utf-8"))+"\n"+str(reply.id)+"\n" for reply in curr_reply_thread)))
                    file_full.flush()

                tweets_processed += 1

                # Unused code for constructing reply tree
                # curr_reply_thread_tree = tweet_helper.construct_reply_thread(curr_tweet)
                # curr_reply_thread_count = count_nodes(curr_reply_thread_tree)

                # print("{} with {} nodes\n".format(curr_reply_thread_tree.data.id, curr_reply_thread_count))
                # print("{}\n".format(curr_reply_thread_tree.__str__()))

                # if curr_reply_thread_count > max_count:
                #     max_count = curr_reply_thread_count
                #     max_node = curr_reply_thread_tree
                #     max_tweet_id = max_node.data.id

                    # file_summary.write("{} with {} nodes\n".format(max_tweet_id, max_count))
                    # file_full.write("{}\n".format(max_node.__str__()))

                    # print("{} with {} nodes\n".format(max_tweet_id, max_count))
                    # print("{}\n".format(max_node.__str__()))

            # Write reply thread length frequency counts to the results_frequency file
            if lines_processed % 10 == 0:
                print("Processed {} lines now with {} tweets".format(lines_processed, tweets_processed))

        file_stats.write('{} lines with {} successfully processed tweets\n'.format(lines_processed, tweets_processed))
        file_stats.flush()
        for count, frequency in sorted(thread_length_freq.items()):
            file_frequency.write('{} - {}\n'.format(count, frequency))
        file_frequency.flush()


def desired_main():

    # Variables
    dir_location = "D:/DLSU/Masters/MS Thesis/data-2016/for_processing"
    start_index_dataset = 1100
    end_index_dataset = 2600
    results_file_name = 'results_{}.txt'.format(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    min_reply_length = 3
    step = 100

    # Load Datasets
    files = FolderIO().get_files(dir_location, False, ".json")
    print("Loaded {} files".format(files.__len__()))

    file_results = open(results_file_name, 'a')
    file_results.write("These are the results for:\n{}\n\n".format(str(files)))

    for index in range(start_index_dataset, end_index_dataset, step):
        # Count replies
        tweet_dataset = JSONParser().parse_files_into_json_generator(files)
        tweet_dataset = itertools.islice(tweet_dataset, index, index+step)
        reply_length_results = TweetUtils().count_replies_list(tweet_dataset)
        print("Counted for {} tweets".format(reply_length_results.__len__()))

        # max reply length
        max_reply_length_result = TweetUtils().reduce_max_reply_length(reply_length_results)

        # frequency count of reply length
        freq_count_reply_lengths = sorted(TweetUtils().frequency_count_reply_lengths(reply_length_results).items())

        tweet_ids_filtered = sorted(list(TweetUtils().filter_reply_lengths_gte(reply_length_results, min_reply_length)), key = itemgetter('reply_length'), reverse = True)
        # full_reply_threads = construct_reply_threads(tweet_ids_with_reply_lengths_filtered)

        # File Writing
        file_results = open(results_file_name, 'a')
        file_results.write('\n\n*****Results from index {} to {} of dataset*****\n'.format(index, index+step))
        file_results.write('\nMax Reply Length:\n{}-{}\n\n'.format(max_reply_length_result['tweet_id'], max_reply_length_result['reply_length']))
        file_results.write('Frequency Count of Reply Lengths:\n')
        for count, frequency in freq_count_reply_lengths:
            file_results.write('{}-{}\n'.format(count, frequency))
        file_results.write('\nTweets with {} thread length:\n'.format(min_reply_length))
        for entry in tweet_ids_filtered:
            tweet_id = entry['tweet_id']
            reply_length = entry['reply_length']
            file_results.write('{}-{}\n'.format(tweet_id, reply_length))

        # file.write(full_reply_threads)
        file_results.flush()

desired_main()

def test_node_printing():
    node1 = Node("1")
    node2 = Node("2")
    node3 = Node("3")
    node4 = Node("4")
    node5 = Node("5")
    node6 = Node("6")
    node7 = Node("7")
    node8 = Node("8")

    node1.add_child(node2)
    node1.add_child(node3)

    node2.add_child(node4)

    node3.add_child(node5)
    node3.add_child(node6)
    node3.add_child(node7)

    node7.add_child(node8)

    print(node1)

    print("Count of nodes is {}".format(count_nodes(node1)))





