import json

class JSONParser:

    # This method assumes that the file contains one valid JSON string per line
    def parse_file_into_json_generator(self, file):
        with file.open() as f:
            return (json.loads(line) for line in f.readlines() if line.strip() != "")

    # This method assumes that the file contains one valid JSON string per line
    def parse_file_into_json_list(self, file):
        with file.open() as f:
            return [json.loads(line) for line in f.readlines()]