from itertools import chain

def parse_files_into_conversation_generator(files):
        generator = iter(())
        for file in files:
            generator = chain(generator, parse_file_into_conversation_generator(file))
        return generator

def parse_file_into_conversation_generator(file):
    with file.open() as f:
        conversation = []
        for line in f.readlines():
            if line.strip() != "":
                line_tokens = line.split()
                entry_dict = {"tweet_id": int(line_tokens[0]), "class": line_tokens[1], "type": line_tokens[2]}
                conversation.append(entry_dict)
            else:
                yield conversation
                conversation = []

        if conversation.__len__() > 0:
            yield conversation
