from data.data_structures import ClassifiedTweet

class ConversationManager:


    conversation_generator = None
    line_generator = None

    def __init__(self, txt_file):
        self.line_generator = self.parse_txt_into_line_generator(txt_file)

    def conversation_generator(self):
        # print("inside")
        conversation = []
        for line in self.line_generator:
            if line.strip() != "":
                # add conversation entry
                tokens = line.split()
                conversation.append(ClassifiedTweet(tokens[0], tokens[1]))
            else:
                yield conversation
                conversation = []

    def parse_txt_into_line_generator(self, file):
        with file.open() as f:
            return (line for line in f.readlines())