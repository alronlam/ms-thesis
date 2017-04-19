import requests
from palmettopy.palmetto import Palmetto

palmetto = Palmetto()

def get_similarity_score(words, coherence_type):
    while(True):
        try:
            return palmetto.get_coherence(words, coherence_type=coherence_type)
        except Exception as e:
            print("Palmetto get similarity score exception: {}".format(e))

# def get_similarity_score(words, coherence_type):
#     while(True):
#         try:
#             url = "http://palmetto.aksw.org/palmetto-webapp/service/{}".format(coherence_type)
#             params = {"words":" ".join(words)}
#             return float(requests.get(url, params).text)
#         except Exception as e:
#             print("Get similarity score exception: {}".format(e))