import pdfminer.high_level
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import re
import pandas as pd
import os

def clean_text(path, spacy_model):

    try:
        text = pdfminer.high_level.extract_text(path)
    
    except:
        return None

    doc = spacy_model(text)
    token_list = []

    # may be helpful if we need any other sentence-specific work later
    for sent in doc.sents:
        
        for token in sent:
            t = token.text.encode(encoding = "ascii", errors = "ignore").decode()

            if t.isalpha() and token.pos_ not in ['X', 'SYM', 'PUNCT']:
                token_list.append(t.lower())

    text = ' '.join(token_list)

    with open('article_dump.bow', 'a', encoding = 'utf-8') as file:
        file.write(text + '\n')

    return text

if __name__ == "__main__":

    nlp = spacy.load("en_core_web_sm")

    print("Processing articles...")

    for i, pdf in enumerate(os.scandir('../downloads/')):
        _ = clean_text(pdf.path, nlp)

        if i % 10 == 0:
            print("Processed {} articles...\n".format(i))

    #x, df = vectorize(sents)
    #print(text)