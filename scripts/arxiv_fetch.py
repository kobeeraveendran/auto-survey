import arxiv
import os
import glob
import argparse

parser = argparse.ArgumentParser(description = 'Fetches a number of documents with the specified topic tag from arXiv. Example usage: python arxiv_fetch.py --topic "text summarization" --num_results 10')

parser.add_argument("-t", "--topic", type = str, nargs = 1, default = "text summarization", help = 'Topic to query arXiv for (surround in quotes if topic includes spaces). Default: "text summarization"')
parser.add_argument("-n", "--num_results", type = int, default = 10, help = "Number of research articles to download from arXiv. Default: 10")

args = parser.parse_args()

results = arxiv.query(query = args.topic, max_results = args.num_results, iterative = True)

for i, paper in enumerate(results()):
    
    paper_info = {
        'title': paper['title'], 
        'url': paper['pdf_url']
    }

    pdf_filename = paper_info['url'].split('/')[-1]
    pdf_dir = "../downloads/"

    os.makedirs(pdf_dir, exist_ok = True)

    arxiv.arxiv.download(paper, dirpath = pdf_dir)