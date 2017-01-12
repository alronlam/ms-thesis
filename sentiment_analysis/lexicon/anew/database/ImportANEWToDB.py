from twitter_data.parsing.csv_parser import CSVParser
from pathlib import Path
from twitter_data.database import DBManager

def create_dict_from_anew_row(row):
    try:
        document = {}
        document["id"] = int(row[0])
        document["english"] = row[1]
        document["filipino"] = row[2].split(",")
        document["valence"] = float(row[3])
        document["arousal"] = float(row[4])
        document["dominance"] = float(row[5])
        return document
    except Exception as e:
        print("AnewCsvParser.create_dict_from_anew_row exception: {}".format(e))
        pass

def import_anew_lexicon_to_db(csv_row_generator):
    document_batch =[]
    for index, row in enumerate(csv_row_generator):
        # print(row)
        document = create_dict_from_anew_row(row)

        if document:
            document_batch.append(document)
            if document_batch.__len__() == 1000: # uses batch size of 1000 before inserting to mongo
                DBManager.add_anew_lexicon_entries(document_batch)
                document_batch = []
                print("Inserted one batch")
    #
    # # insert the last batch that might have not reached 100 in number
    if document_batch.__len__() > 0 :
        DBManager.add_anew_lexicon_entries(document_batch)


def driver_import_code():
    anew_csv_file = Path('C:/Users/user/PycharmProjects/ms-thesis/sentiment_analysis/lexicon/anew/ANEW Lexicon with Filipino Translation.csv')
    csv_row_generator = CSVParser.parse_file_into_csv_row_generator(anew_csv_file, True)
    import_anew_lexicon_to_db(csv_row_generator)