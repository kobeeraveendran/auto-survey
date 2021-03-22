import pdfminer.high_level
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re

def clean_text(path, spacy_model):
    '''
    path (str): path to PDF file from which text is extracted
    '''

    # extract text from pdf @ path
    text = pdfminer.high_level.extract_text(path)

    doc = spacy_model(text)

    # sentence segmentation
    assert doc.has_annotation("SENT_START")

    sentences = []

    for sentence in doc.sents:
        s = re.sub("['|.|,|0-9]", '', sentence.text)
        s = s.encode(encoding = "ascii", errors = "ignore").decode()
        sentences.append(s.lower())
        #sentences.append(re.sub("['|.|,|0-9]", '', sentence))

    for i in range(5):
        print(sentences[i])

    # text = text.split(' ')
    # text = [s.lower() for s in text if s.isalpha()]

    # non-eng removal
    # for i, t in enumerate(text):
    #     text[i] = t.encode(encoding = "ascii", errors = "ignore").decode()

    #text = ' '.join(text)

    #token_list = [token.text for token in tokens if str(token).isalpha() or token.pos_ not in ['X', 'SYM', 'PUNCT']]

    #text = ' '.join(token_list)

    # with open('test.txt', 'w', encoding = 'utf-8') as file:
    #     file.write(text)

    return text

def vectorize(text, spacy_model):
    sentences = spacy_model(text)

if __name__ == "__main__":

    doc = "../downloads/1201.2240v1.Bengali_text_summarization_by_sentence_extraction.pdf"
    
    nlp = spacy.load("en_core_web_sm")
    text = clean_text(doc, nlp)
    #print(text)