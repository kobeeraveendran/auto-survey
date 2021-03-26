import pdfminer.high_level
import spacy
import string

def pdf2text(path):
    text = pdfminer.high_level.extract_text(path)

    # basic cleaning
    text = text.split(' ')
    text = [s for s in text if s.isalpha()]

    for i, t in enumerate(text):
        text[i] = t.encode(encoding = "ascii", errors = "ignore").decode()

    text = ' '.join(text)

    tokenizer = spacy.load("en_core_web_sm")
    tokens = tokenizer(text)

    token_list = [token.text for token in tokens if str(token).isalpha() or token.pos_ not in ['X', 'SYM', 'PUNCT']]

    text = ' '.join(token_list)

    with open('test.bow', 'w', encoding = 'utf-8') as file:
        file.write(text)

    return text

if __name__ == "__main__":

    doc = "../downloads/1201.2240v1.Bengali_text_summarization_by_sentence_extraction.pdf"
    
    text = pdf2text(doc)
    #print(text)