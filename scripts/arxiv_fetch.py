import arxiv
import os
import glob

results = arxiv.query(query = "text summarization", max_results = 400, iterative = True)

for i, paper in enumerate(results()):
    
    paper_info = {
        'title': paper['title'], 
        'url': paper['pdf_url']
    }

    pdf_filename = paper_info['url'].split('/')[-1]
    pdf_dir = "../downloads/"

    os.makedirs(pdf_dir, exist_ok = True)

    arxiv.arxiv.download(paper, dirpath = pdf_dir)