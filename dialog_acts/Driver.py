from twitter_data.parsing.folders.FolderIO import FolderIO
from dialog_acts.ConversationManager import ConversationManager
from dialog_acts.TSVParser import TSVParser


# get all tsv files within the directory
def filter_tsv_conversations():
    tsv_files = FolderIO().get_files('D:/DLSU/Masters/MS Thesis/data-2016/Context-Based_Tweets', True, '.tsv')
    file_results = open('conversations.txt', 'w')

    tsv_parser = TSVParser(tsv_files)

    curr_conversation = tsv_parser.next_conversation()

    while curr_conversation:
        if curr_conversation.__len__() > 1:
            # print("\n".join(", ".join(entry) for entry in curr_conversation) + "\n")
            file_results.write("\n".join(" ".join(entry) for entry in curr_conversation) + "\n\n")
        else:
            pass

        curr_conversation = tsv_parser.next_conversation()

    file_results.flush()


def main():
    conversations_txt_file = FolderIO().get_files('conversations/', False, '.txt')[0]
    conversation_manager = ConversationManager(conversations_txt_file)
    conversation_generator = conversation_manager.conversation_generator()
    for conversation in conversation_generator:
        print(conversation)
        input()

main()