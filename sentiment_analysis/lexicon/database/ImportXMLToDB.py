from twitter_data.database import DBManager

def import_lexicon_to_db(xml_row_generator):
    document_batch =[]
    for row in xml_row_generator:

        document = {}
        for child in row:
            document[child.tag.lower()] = int(child.text) if child.tag.lower() == 'id' else child.text

        document_batch.append(document)

        if document_batch.__len__() == 1000: # uses batch size of 1000 before inserting to mongo
            DBManager.add_lexicon_so_entries(document_batch)
            document_batch = []
            print("Inserted one batch")

    # insert the last batch that might have not reached 100 in number
    if document_batch.__len__() > 0 :
        DBManager.add_lexicon_so_entries(document_batch)


# import_lexicon_to_db(LexiconXMLParser.parse_xml_file_into_row_generator('C:/Users/user/PycharmProjects/ms-thesis/xml_parser/subj-lex-product.xml'))

