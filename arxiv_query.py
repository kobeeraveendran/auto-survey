import arxiv
import pdfminer.high_level

import os
import glob

results = arxiv.query(query = "text summarization", max_results = 5, iterative = True)

for i, paper in enumerate(results()):
    
    # paper details
    #print(paper)

    paper_info = {
        'title': paper['title'], 
        'url': paper['pdf_url']
    }

    pdf_filename = paper_info['url'].split('/')[-1]

    print(paper_info)

    os.makedirs('./downloads', exist_ok = True)

    # download pdf versions
    #arxiv.arxiv.download(paper, dirpath = './downloads/')
    pdf_file = glob.glob(os.path.join('downloads', '{}*'.format(pdf_filename)))[0]
    print("pdf filename: ", pdf_file)
    text = pdfminer.high_level.extract_text(pdf_file)
    
    if i == 0:
        print(text)