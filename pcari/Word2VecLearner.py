import csv
from scipy import sparse


def load_csv_dataset(file_path):
    with open(file_path, "r", encoding="utf8", newline="") as csv_file:
        row_reader = csv.reader(csv_file, delimiter=',')
        dataset = [row for row in row_reader]

    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    return (x,y)


(x,y) = load_csv_dataset("")


# import gensim
# # let X be a list of tokenized texts (i.e. list of lists of tokens)
# model = gensim.models.Word2Vec(x, size=100)
# w2v = dict(zip(model.index2word, model.syn0))


import numpy as np
def build_cooccur(vocab, corpus, window_size=10, min_count=None):
    vocab_size = len(vocab)
    id2word = dict((i, word) for word, (i, _) in vocab.iteritems())

    # Collect cooccurrences internally as a sparse matrix for
    # passable indexing speed; we'll convert into a list later
    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                      dtype=np.float64)

    for i, line in enumerate(corpus):
        tokens = line.strip().split()
        token_ids = [vocab[word][0] for word in tokens]

    for center_i, center_id in enumerate(token_ids):
        # Collect all word IDs in left window of center word
        context_ids = token_ids[max(0, center_i - window_size) : center_i]
        contexts_len = len(context_ids)

    for left_i, left_id in enumerate(context_ids):
        # Distance from center word
        distance = contexts_len - left_i

        # Weight by inverse of distance between words
        increment = 1.0 / float(distance)

        # Build co-occurrence matrix symmetrically (pretend
        # we are calculating right contexts as well)
        cooccurrences[center_id, left_id] += increment
        cooccurrences[left_id, center_id] += increment
