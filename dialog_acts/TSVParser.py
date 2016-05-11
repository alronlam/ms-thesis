from itertools import chain

class TSVParser:

    tsv_line_generator = None

    def __init__(self, files):
        self.tsv_line_generator = self.parse_files_into_line_generator(files)

    def next_conversation(self):
        conversation = []
        try:
            tsv_line = next(self.tsv_line_generator)
            while tsv_line.strip() != "":
                conversation.append(tsv_line.split())
                try:
                    tsv_line = next(self.tsv_line_generator)
                except:
                    pass
            return conversation
        except:
            return None

    def parse_files_into_line_generator(self, files):
        generator = iter(())
        for file in files:
            generator = chain(generator, self.parse_file_into_line_generator(file))
        return generator

    def parse_file_into_line_generator(self, file):
        with file.open() as f:
            return (line for line in f.readlines())