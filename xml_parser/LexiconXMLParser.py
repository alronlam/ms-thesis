import xml.etree.ElementTree as ET

# This assumes the ff. format:
# <ROWSET>
# 	<ROW>
# 		<ID>1</ID>
# 		<ENG>abash</ENG>
# 		<FIL>tarantahin</FIL>
# 		<polarity>negative</polarity>
# 	</ROW>
# </ROWSET>
def parse_xml_file_into_row_generator(file_name):
    tree = ET.parse(file_name)
    root = tree.getroot()
    for child in root.iter("ROW"):
        yield child


