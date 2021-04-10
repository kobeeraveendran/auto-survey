# create environment from file and activate it
# if you run into issues, using environment_verbose.yml may help for finding missing packages
conda env create --file ../environment.yml

conda activate autosurvey

# download a small set of research papers
# additional command line args available; to view, run as: python arxiv_fetch.py -h
python arxiv_fetch.py

# download and install spaCy language model
python -m spacy download en_core_web_sm

# preprocess the downloaded documents
cd ../summarize/ && python preprocess.py