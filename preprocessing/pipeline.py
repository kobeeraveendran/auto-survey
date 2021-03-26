


def load_bow(path="", vocab=None):
    word_dict = dict()

    def add_word(word=""):
        word = word.lower()
        if word in word_dict:
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





if __name__ == "__main__":

    data_path = "./test.bow"
    raw_bow = load_bow(path=data_path)
    print("Loaded: ", raw_bow)