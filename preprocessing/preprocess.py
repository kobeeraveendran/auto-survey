import pdfminer.high_level
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import pandas as pd

def clean_text(path, spacy_model):

    text = pdfminer.high_level.extract_text(path)

    doc = spacy_model(text)
    token_list = []

    # may be helpful if we need any other sentence-specific work later
    for sent in doc.sents:
        
        for token in sent:
            t = token.text.encode(encoding = "ascii", errors = "ignore").decode()

            if t.isalpha() and token.pos_ not in ['X', 'SYM', 'PUNCT']:
                token_list.append(t.lower())

    text = ' '.join(token_list)

    with open('test.bow', 'w', encoding = 'utf-8') as file:
        file.write(text)

    return text

if __name__ == "__main__":

    doc = "../downloads/1201.2240v1.Bengali_text_summarization_by_sentence_extraction.pdf"
    
    nlp = spacy.load("en_core_web_sm")
    text = clean_text(doc, nlp)

    #x, df = vectorize(sents)
    #print(text)