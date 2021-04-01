from gensim import models
from gensim.corpora import Dictionary
from collections import defaultdict

import os

# Can use max count to limit impact of frequent words by capping their count
def load_bow(path="", vocab=None, max_count=0):
    word_dict = dict()

    def add_word(word=""):
        if word in word_dict:
            if (max_count == 0) or (word_dict[word] < max_count):
                word_dict[word] = word_dict[word] + 1
        else:
            word_dict[word] = 1

    if path:
        with open(path, 'r') as file:
            for line in file:
                for word in line.split():

                    # No vocab, include all words
                    if vocab == None:
                        add_word(word)

                    # Otherwise, only include words in vocab
                    # Also, when using vocab, use the word id instead of the word
                    elif word in vocab.token2id:
                        add_word(word)

    else:
        for word in 

    return word_dict

def convert_bow(bow=dict()):
    words = []
    for word in bow:
        for i in range(bow[word]):
            words.append(word)

    return words


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

    words = []
    for word in vocab:
        words.append(word)
    words = [words]

    #print ("WORDS: ", words)
    dct = Dictionary(words)

    return dct



if __name__ == "__main__":

    ### TRAINING LDA MODEL ###
    # Load Data

    overall_model = models.LdaModel(
        num_topics = 15, 
        alpha = "asymmetric", 
        eta = "auto", 
        minimum_probability = 0.02, 
        per_word_topics = True
    )
    
    overall_bow = []

    # individual docs
    for doc_path in os.scandir("../bags/"):
        raw_bow = load_bow(path = doc_path)
        doc_vocabulary = create_vocabulary([raw_bow])
        overall_bow.append(raw_bow)

        doc_filtered_bow = load_bow(path = doc_path, vocab = doc_vocabulary)
        train_data = [[]]

        for word in doc_vocabulary.token2id.keys():
            train_data[0].append(word)

        train_data = [doc_vocabulary.doc2bow(convert_bow(doc_filtered_bow))]

        doc_model = models.LdaModel(
            corpus = train_data, 
            num_topics = 10, 
            alpha = "asymmetric", 
            eta = "auto", 
            minimum_probability = 0.02, 
            per_word_topics = True
        )

        overall_model.update(corpus = train_data, update_every = 1)
        doc_topic_dist = doc_model.get_document_topics(doc_vocabulary.doc2bow(convert_bow(doc_filtered_bow)))
    
    overall_vocabulary = create_vocabulary(overall_bow)
    overall_topic_dist = overall_model.get_document_topics(overall_vocabulary.doc2bow(convert_bow()))

    # Save LDA model
    # Note: Also need to save word_to_id for inputting data into model
    overall_model.save("./overall_model.lda")
