import pdfminer.high_level
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import pandas as pd

def clean_text(path, spacy_model):
    '''
    input:
        - path (str): path to PDF file from which text is extracted
        - spacy_model: loaded spaCy language model (english small model - 'en_core_web_sm')
    output:
        - sentences: list of strings, where each element is a cleaned sentence from the PDF
    '''

    # extract text from pdf @ path
    text = pdfminer.high_level.extract_text(path)

    doc = spacy_model(text)

    # sentence segmentation
    assert doc.has_annotation("SENT_START")

    sentences = []

    # preserve sentences for reconstruction later
    for sentence in doc.sents:
        s = re.sub("et. al|['.,0-9]", '', sentence.text)
        s = s.encode(encoding = "ascii", errors = "ignore").decode()
        sentences.append(s.lower())

    # with open('test.txt', 'w', encoding = 'utf-8') as file:
    #     file.writelines(sentences)
        

    return sentences

# TODO: add further preprocessing as per the steps outlined in the milestone report (i.e. pruning words by high frequency); the stuff here is just general preprocessing
def vectorize(docs):

    '''
    input:
        - docs: list of strings, where each element is a sentence (provided by clean_text())

    output:
        - x: matrix of word frequencies (num. occurrences x vocab. size) to be fed into LDA
        - df: pandas DataFrame containing the same matrix as in x, with feature names (word strings) included in header
    '''

    vect = CountVectorizer(stop_words = "english")
    x = vect.fit_transform(docs)

    df = pd.DataFrame(x.toarray(), columns = vect.get_feature_names())

    return x, df

if __name__ == "__main__":

    doc = "../downloads/1201.2240v1.Bengali_text_summarization_by_sentence_extraction.pdf"
    
    nlp = spacy.load("en_core_web_sm")
    sents = clean_text(doc, nlp)

    x, df = vectorize(sents)
    #print(text)