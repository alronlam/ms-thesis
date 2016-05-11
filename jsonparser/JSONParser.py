from itertools import chain
import json

class JSONParser:

    def parse_files_into_json_generator(self, files):
        generator = iter(())
        for file in files:
            generator = chain(generator, self.parse_file_into_json_generator(file))
        return generator

    # This method assumes that the file contains one valid JSON string per line
    def parse_file_into_json_generator(self, file):
        with file.open() as f:
            return (json.loads(line) for line in f.readlines() if line.strip() != "")

    # This method assumes that the file contains one valid JSON string per line
    def parse_file_into_json_list(self, file):
        with file.open() as f:
            return [json.loads(line) for line in f.readlines()]