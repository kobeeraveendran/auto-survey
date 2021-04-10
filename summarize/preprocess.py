import pdfminer.high_level
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import re
import pandas as pd
import os

def clean_text(id, path, spacy_model):

    try:
        text = pdfminer.high_level.extract_text(path)
    
    except:
        return None

    #doc = spacy_model(text)
    token_list = []
    sentences = []

    # for sent in doc.sents:

    #     curr_sent = []

    for sent in text.split('.'):
        doc = spacy_model(sent)

        curr_sent = []

        for token in doc:
            t = token.text.encode(encoding = "ascii", errors = "ignore").decode()
            t = t.lower()

            if t.isalpha() and token.pos_ not in ['X', 'SYM', 'PUNCT']:
                if t not in spacy_model.Defaults.stop_words and len(t) > 2:
                    token_list.append(t)
                curr_sent.append(t)

        sentences.append(' '.join(curr_sent) + '\n')

    text = ' '.join(token_list)

    id_line = "{}:{}\n".format(id, path.split('/')[-1])

    with open('../bags/{}.bow'.format(id), 'w', encoding = 'utf-8') as file:
        file.write(text + '\n')

    with open("../sentences/{}.sentences".format(id), 'w', encoding = 'utf-8') as file:
        file.writelines(sentences)

    return id_line

if __name__ == "__main__":

    nlp = spacy.load("en_core_web_sm")

    os.makedirs("../bags/", exist_ok = True)
    os.makedirs("../sentences/", exist_ok = True)

    print("Processing documents...")

    id_map = []

    print("Processed documents: 0", end = '\r')

    for i, pdf in enumerate(os.scandir('../downloads/')):
        id = clean_text(i, pdf.path, nlp)
        id_map.append(id)

        print("Processed documents:", i + 1, end = '\r')

    print()

    with open("ids.txt", 'w', encoding = 'utf-8') as file:
            file.writelines(id_map)

    #x, df = vectorize(sents)
    #print(text)