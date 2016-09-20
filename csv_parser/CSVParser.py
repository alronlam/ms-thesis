from itertools import chain
import csv
import os
class CSVParser:

    @staticmethod
    def parse_file_into_csv_row_generator(file):
        with open(file.absolute().__str__(), newline='') as csv_file:
            row_reader = csv.reader(csv_file, delimiter=',')

            for row in CSVParser.iterator_wrapper(row_reader):
                yield_row = [x for x in row if x.strip() != "" ]
                if yield_row.__len__() > 0:
                    yield yield_row

    @staticmethod
    def iterator_wrapper(generator):
        while True:
            try:
              yield next(generator)
            except StopIteration:
              raise
            except Exception as e:
              print(e) # or whatever kind of logging you want
              pass

    @staticmethod
    def parse_files_into_csv_row_generator(files):
        generator = iter(())
        for file in files:
            generator = chain(generator, CSVParser.parse_file_into_csv_row_generator(file))
        return generator