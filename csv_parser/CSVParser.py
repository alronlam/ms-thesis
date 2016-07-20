import csv
class CSVParser:

    @staticmethod
    def parse_file_into_csv_row_generator(file_name):
     with open('../'+file_name, newline='') as csv_file:
         row_reader = csv.reader(csv_file, delimiter=',')
         for row in row_reader:
            yield_row = [x for x in row if x.strip() != "" ]
            if yield_row.__len__() > 0:
                yield yield_row
