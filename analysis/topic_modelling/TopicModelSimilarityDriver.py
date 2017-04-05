from sentiment_analysis.preprocessing import PreProcessing
from twitter_data.parsing.folders import FolderIO


def load_community_docs(dir):
    txt_files = FolderIO.get_files(dir, False, '.txt')
    community_docs = []
    for txt_file in txt_files:
        with txt_file.open(encoding="utf-8") as f:
            community_docs.append([line.strip() for line in f.readlines() if line.strip()])
    return community_docs

def preprocess_docs(docs):
    preprocessors = []
    return PreProcessing.preprocess_strings(docs, preprocessors)


def generate_topic_model(doc):
    pass

def generate_topic_similarities(topic_models):
    pass

def save_topic_similarities(topic_similarities):
    pass



community_docs = load_community_docs("C:/Users/user/PycharmProjects/ms-thesis/community_detection/Experiments Round 2 - Mention-based - Revised/Brexit/Mentions/Texts")
community_topic_models = [generate_topic_model(community_doc) for community_doc in community_docs]
topic_similarities = generate_topic_similarities(community_topic_models)
save_topic_similarities(topic_similarities)