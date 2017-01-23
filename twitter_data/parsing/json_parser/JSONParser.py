from itertools import chain
import json

def parse_files_into_json_generator(files):
    generator = iter(())
    for file in files:
        generator = chain(generator, parse_file_into_json_generator(file))
    return generator

# This method assumes that the file contains one valid JSON string per line
def parse_file_into_json_generator(file):
    with file.open() as f:
        for line in f.readlines():
            if line.strip() != "":
                try:
                    yield json.loads(line)
                except Exception as e:
                    print(e)

        # return (json.loads(line) for line in f.readlines() if line.strip() != "")


# This method assumes that the file contains one valid JSON string per line
def parse_file_into_json_list(file):
    with file.open() as f:
        return [json.loads(line) for line in f.readlines()]