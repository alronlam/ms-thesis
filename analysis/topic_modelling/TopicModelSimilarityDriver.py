def load_community_docs(dir):
    pass

def generate_topic_model(doc):
    pass

def generate_topic_similarities(topic_models):
    pass

def save_topic_similarities(topic_similarities):
    pass

community_docs = load_community_docs(dir)
community_topic_models = [generate_topic_model(community_doc) for community_doc in community_docs]
topic_similarities = generate_topic_similarities(community_topic_models)
save_topic_similarities(topic_similarities)


