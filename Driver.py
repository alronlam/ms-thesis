from FolderIO import FolderIO
from JSONParser import JSONParser
from TweetHelper import TweetHelper
from Node import Node
import datetime


def count_nodes(node):
    count = 0
    if node is not None:
        count += 1
    for child in node.children:
        count += count_nodes(child)
    return count


def main():
    folderIO = FolderIO()
    files = folderIO.get_files("D:/DLSU/Masters/MS Thesis/data-2016/02/", False, ".json")

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
        tweet_helper = TweetHelper()
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

main()


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





