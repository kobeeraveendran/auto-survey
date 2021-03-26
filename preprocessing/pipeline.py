from gensim.models import LdaModel
from collections import defaultdict

# Can use max count to limit impact of frequent words by capping their count
def load_bow(path="", vocab=None, max_count=0):
    word_dict = dict()

    def add_word(word=""):
        if word in word_dict:
            if (max_count == 0) or (word_dict[word] < max_count):
                word_dict[word] = word_dict[word] + 1
        else:
            word_dict[word] = 1

    with open(path, 'r') as file:
        for line in file:
            for word in line.split():

                # No vocab, include all words
                if vocab == None:
                    add_word(word)

                # Otherwise, only include words in vocab
                elif word in vocab:
                    add_word(word)

    return word_dict

def create_vocabulary(documents=[], min_word_frequency=0.05, max_word_frequency=1, min_word_count=5):
    if len(documents) < 1:
        return defaultdict(bool)

    # Collect word statistics
    word_counts = defaultdict(int)
    word_appearances = defaultdict(int)

    for doc in documents:
        seen_words = defaultdict(bool)

        for word in doc:
            word_counts[word] += doc[word]

            if word not in seen_words:
                word_appearances[word] += 1
                seen_words[word] = True

    # Build vocabulary
    vocab = defaultdict(bool)
    for word in word_counts:
        freq = word_appearances[word] / len(documents)
        if (word_counts[word] >= min_word_count) and (freq >= min_word_frequency and freq <= max_word_frequency):
            vocab[word] = True

    return vocab






if __name__ == "__main__":

    data_path = "./test.bow"
    raw_bow = load_bow(path=data_path)

    vocabulary = create_vocabulary([raw_bow])

    print("Vocab: ", vocabulary.keys())

    filtered_bow = load_bow(path=data_path, vocab=vocabulary)

    print("Filtered BOW: ", filtered_bow)

