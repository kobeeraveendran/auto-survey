import pdfminer.high_level



def pdf2text(path):
    text = pdfminer.high_level.extract_text(path)

    # basic cleaning
    text = text.split(' ')
    text = [s for s in text if s.isalpha()]
    text = ' '.join(text)

    with open('test.txt', 'w', encoding = 'utf-8') as file:
        file.write(text)

    return text

if __name__ == "__main__":

    doc = "../downloads/1201.2240v1.Bengali_text_summarization_by_sentence_extraction.pdf"
    
    text = pdf2text(doc)
    print(text)