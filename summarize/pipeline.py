from gensim import models
from gensim.corpora import Dictionary
from collections import defaultdict
from rouge_score import rouge_scorer


import os, math, argparse, time

# for summary comparison
from gensim.summarization.summarizer import summarize

# Can use max count to limit impact of frequent words by capping their count
def load_bow(path="", existing_bow = None, vocab=None, max_count=0, min_word_len=2, as_list=False):

    word_dict = dict()
    if as_list:
        word_dict = []
        

    def add_word(word=""):
        if as_list:
            word_dict.append(word)

        else:
            if word in word_dict:
                if (max_count == 0) or (word_dict[word] < max_count):
                    word_dict[word] = word_dict[word] + 1
            else:
                word_dict[word] = 1

    if path:
        with open(path, 'r') as file:
            for line in file:
                for word in line.split():
                    word = word.lower().strip()

                    # Enforce minimum word length
                    if (len(word) < min_word_len):
                        continue

                    # No vocab, include all words
                    if vocab == None:
                        add_word(word)

                    # Otherwise, only include words in vocab
                    # Also, when using vocab, use the word id instead of the word
                    elif word in vocab.token2id:
                        add_word(word)

    else:
        # filter a pre-existing bow as specified by the vocabulary
        if existing_bow:
            for key in existing_bow:
                if key in vocab.token2id:
                    if key in word_dict:
                        word_dict[key] += existing_bow[key]

                    else:
                        word_dict[key] = existing_bow[key]

    return word_dict

# Returns list of Strings when include original is False
# Returns list of tuple (filtered sentence string, complete sentence string) when include original is True
def load_sentences(path="", vocab=None, include_original=True):
    sentences = []

    if path:
        with open(path, 'r') as file:
            for line in file:
                clean_line = ""
                filtered_line = ""

                for word in line.split():
                    word = word.lower().strip()

                    if (clean_line == ""):
                        clean_line = word
                    
                    else:
                        clean_line = clean_line + " " + word

                    if vocab and (word in vocab.token2id):
                        if (filtered_line == ""):
                            filtered_line = word

                        else:
                            filtered_line = filtered_line + " " + word

                if (include_original):
                    sentences.append((filtered_line, clean_line))
        
                else:
                    if vocab:
                        sentences.append(filtered_line)
                    else:
                        sentences.append(clean_line)

    return sentences

def convert_bow(bow=dict()):
    words = []
    for word in bow:
        for i in range(bow[word]):
            words.append(word)

    return words

# Returns a gensim.corpora.Dictionary 
def create_vocabulary(documents=[], min_word_frequency=0.05, max_word_frequency=1, min_word_count=5):
    if len(documents) < 1:
        return defaultdict(bool)

    # Collect word statistics
    word_counts = defaultdict(int)
    word_appearances = defaultdict(int)

    for doc in documents:
        seen_words = defaultdict(bool)

        for word in doc:
            word_counts[word] += doc[word]

            if word not in seen_words:
                word_appearances[word] += 1
                seen_words[word] = True

    # Build vocabulary
    vocab = defaultdict(bool)
    for word in word_counts:
        freq = word_appearances[word] / len(documents)
        if (word_counts[word] >= min_word_count) and (freq >= min_word_frequency and freq <= max_word_frequency):
            vocab[word] = True

    words = []
    for word in vocab:
        words.append(word)
    words = [words]

    #print ("WORDS: ", words)
    dct = Dictionary(words)

    return dct



if __name__ == "__main__":

    # track execution time
    start_time = time.time()

    # cmdline args
    parser = argparse.ArgumentParser(description = "Summarizes a set of research articles using LDA.")

    parser.add_argument(
        "--all", 
        action = "store_true", 
        help = "Whether to generate summaries for every document used during training. Include this flag to use all docs, or exclude to use only those specified in targets.txt. \
        This flag may be useful for reproducing the results shown in our plots. Example usage: python summarize.py --all"
    )
    parser.add_argument(
        "--num_topics", 
        type = int, 
        default = 200, 
        help = "Number of topics expected across all documents for LDA training (choose relative to the number of documents you are training on). Default 200"
    )

    args = parser.parse_args()

    ### TRAINING LDA MODEL ###
    # Load Data
    
    doc_bows = []
    overall_bow = {}

    topic_dists = []

    VERBOS = 1

    BOW_DIR = "../bags/"
    SENTENCES_DIR = "../sentences/"



    # --- Build Vocabulary from entire corpus ---

    print("Loading raw BOW Files...", end="\r")
    # Load every document into memory
    total_files = len(os.listdir(BOW_DIR))
    for i, doc_path in enumerate(os.scandir(BOW_DIR)):
        raw_bow = load_bow(path = doc_path)
        doc_bows.append(raw_bow)

        # Progress tracing
        if (i + 1) % 17 == 0:
            print("Loading BOW Files... ", i, "/", total_files, end="\r")

    print("Loading BOW Files... Complete              ")

    # Build vocabulary using words that:
    # * Appear in less than 60% of docs
    # * Appear at least 100 times
    print("Building Vocabulary...", end="\r")
    model_vocab = create_vocabulary(doc_bows, min_word_frequency=0, max_word_frequency=0.6, min_word_count=400)
    print("Building Vocabulary... Complete              ")

    if (VERBOS):
        print("Generated Vocabulary with ", len(model_vocab.token2id.keys()), " words")
        if (VERBOS == 2): 
            print(model_vocab.token2id.keys())

    # Free raw documents from memory
    doc_bows = []




    # --- Train LDA Model over entire corpus ---
    NUM_TOPICS = args.num_topics
    BATCH_SIZE = 16
    MAX_ITERATIONS = 200

    lda_model = models.LdaModel(
        num_topics = NUM_TOPICS,
        id2word=model_vocab,
        alpha = "asymmetric", 
        # eta = "auto", 
        minimum_probability = 0.02, 
        per_word_topics = True
    )

    print("Loading filtered BOW Files...", end="\r")
    doc_bows = []
    for i, doc_path in enumerate(os.scandir(BOW_DIR)):
        raw_bow = load_bow(path = doc_path, vocab = model_vocab, as_list=True)
        doc_bows.append(model_vocab.doc2bow(raw_bow))

        # Progress tracing
        if (i + 1) % 17 == 0:
            print("Loading filtered BOW Files... ", i, "/", total_files, end="\r")
    print("Loading filtered BOW Files... Complete              ")

    print("Training LDA Model...", end="\r")
    doc_itr = enumerate(doc_bows)

    # Train on dataset in batches
    total_iters = 0
    while True:
        total_iters += 1
        batch = []
        try:
            for i in range(BATCH_SIZE):
                batch.append(next(doc_itr)[1])
        except StopIteration:
            pass

        print("Training LDA Model... ", BATCH_SIZE, "/", total_files, end="\r")

        if VERBOS == 2:
            print("Training LDA Model... ", BATCH_SIZE, "/", total_files)

            print("Current Batch: ", batch)
            print("len: ", len(batch))
            print("t: ", type(batch[0]))

        if len(batch) < 1:
            break

        lda_model.update(corpus=batch,
            iterations=MAX_ITERATIONS,
            update_every=0,
            gamma_threshold=0.005)
        
    print("Training LDA Model... Complete (", total_iters, "batches of", BATCH_SIZE, ")                       ")




    # --- Extract summary of target documents  ---
    TARGETS_FILE = "./targets.txt"
    PER_DOC_SENTENCES = 5
    MIN_SENTENCE_LENGTH = 5

    print("Loading Summary Targets...", end="\r")
    targets = []
    overall_bow = []
    overall_sentences = []

    # if all documents used for training should also have summaries generated for them
    # TODO: maybe clean up this duplication later
    if args.all:
        
        for d_bow, d_sent in zip(os.scandir(BOW_DIR), os.scandir(SENTENCES_DIR)):

            target_bow = load_bow(path = d_bow.path, vocab=model_vocab, as_list=True)

            for word in target_bow:
                overall_bow.append(word)

            target_bow = model_vocab.doc2bow(raw_bow)
            target_sentences = load_sentences(path = d_sent.path, vocab=model_vocab, include_original=True)

            for sentence in target_sentences:
                overall_sentences.append(sentence)

            targets.append((target_bow, target_sentences))
            
            # target_bow = load_bow(path = d_bow.path, vocab = model_vocab, as_list = True)

            # for word in target_bow:
            #     overall_bow.append(word)

            # target_bow = model_vocab.doc2bow(raw_bow)
            # target_sentences = load_sentences(path = d_sent.path)

            # for sentence in target_sentences:
            #     overall_sentences.append(sentence)

            # targets.append((target_bow, target_sentences))
    
    # otherwise, draws the pool of documents to generate summaries for from `targets.txt`
    else:
        with open(TARGETS_FILE, 'r') as file:
            for line in file:
                    for word in line.split():
                        target_id = int(word.lower().strip())
                        
                        target_bow = load_bow(path=BOW_DIR+str(target_id)+".bow", vocab=model_vocab, as_list=True)

                        for word in target_bow:
                            overall_bow.append(word)

                        target_bow = model_vocab.doc2bow(raw_bow)
                        target_sentences = load_sentences(path=SENTENCES_DIR+str(target_id)+".sentences", vocab=model_vocab, include_original=True)

                        for sentence in target_sentences:
                            overall_sentences.append(sentence)

                        targets.append((target_bow, target_sentences))

    #print("\n\nDEBUG:\n\n", targets[0])

    overall_bow = model_vocab.doc2bow(overall_bow)

    print("Loading Summary Targets... Complete              ",)

    doc_summaries = []
    print("Generating per document summaries...", end="\r")
    for i in range(len(targets)):
        print("Generating per document summaries... ", i, "/", len(targets), end="\r")

        # Extract document topic distribution
        doc_bow = targets[i][0]
        doc_sentences = targets[i][1]

        doc_topics = lda_model.get_document_topics(doc_bow, minimum_probability=0)

        # Extract per document summaries as most probable setences based on topic distribution
        best_sentences = [("", -1, -1)] * PER_DOC_SENTENCES
        idx = 0
        for sentence in doc_sentences:
            score = 0
            word_topics = []
            idx = idx + 1

            if (len(sentence[0]) < 2):
                continue

            if (len(sentence[0].split()) < MIN_SENTENCE_LENGTH):
                continue

            for word in sentence[0].split():
                word = word.strip()
                if word in model_vocab.token2id:
                    word_topics.append(lda_model.get_term_topics(model_vocab.token2id[word], minimum_probability=0))

            for topic_id in range(NUM_TOPICS):
                topic_score = 1
                topic_prob = 0

                for topic in doc_topics:
                    if topic[0] == topic_id:
                        topic_prob = topic[1]

                for word in word_topics:
                    word_prob = 0
                    for topic in word:
                        if topic[0] == topic_id:
                            word_prob = topic[1]
                    topic_score = topic_score * word_prob * topic_prob

                score = score + topic_score

            for i in range(PER_DOC_SENTENCES):
                if score > best_sentences[i][1]:
                    best_sentences[i] = (sentence[1], score, idx)
                    break


        sorted_summary = sorted(best_sentences, key=lambda x: x[-1])
        doc_summary = []
        for sentence in sorted_summary:
            doc_summary.append(sentence)

        doc_summaries.append(doc_summary)

    print("Generating per document summaries... Compelete                    ")

    if VERBOS: #and not args.all:
        print("Generated Summaries:")
        for i, doc in enumerate(doc_summaries):
            print("    Summary of document", i, ":")
            for sentence in doc:
                print("    ", sentence[0], " [", sentence[1], "]")
            print("")

    print("Generating multi-document summary...", end="\r")

    OVERALL_SENTENCES = 10

    doc_topic = lda_model.get_document_topics(overall_bow, minimum_probability=0)

    best_sentences = [("", -1, -1)] * OVERALL_SENTENCES
    idx = 0
    for sentence in overall_sentences:
        score = 0
        word_topics = []
        idx = idx + 1

        if (len(sentence[0]) < 2):
            continue

        if (len(sentence[0].split()) < MIN_SENTENCE_LENGTH):
            continue

        for word in sentence[0].split():
            word = word.strip()
            if word in model_vocab.token2id:
                word_topics.append(lda_model.get_term_topics(model_vocab.token2id[word], minimum_probability=0))

        for topic_id in range(NUM_TOPICS):
            topic_score = 1
            topic_prob = 0

            for topic in doc_topic:
                if topic[0] == topic_id:
                    topic_prob = topic[1]

            for word in word_topics:
                word_prob = 0
                for topic in word:
                    if topic[0] == topic_id:
                        word_prob = topic[1]
                topic_score = topic_score * word_prob * topic_prob

            score = score + topic_score

        for i in range(OVERALL_SENTENCES):
            if score > best_sentences[i][1]:
                best_sentences[i] = (sentence[1], score, idx)
                break

    sorted_summary = sorted(best_sentences, key=lambda x: x[-1])
    overall_summary = []
    for sentence in sorted_summary:
        overall_summary.append(sentence)

    print("Generating multi-document summary... Complete")

    overall_summary_text = []
    if VERBOS:
        print("Overall Summary:")
        for sentence in overall_summary:
            print("    ", sentence[0], " [", sentence[1], "]")
            overall_summary_text.append(sentence[0])
        print("")

    overall_summary_text = '. '.join(overall_summary_text)

    print("Alternative multi-doc summary:\n")

    overall_summary_alt = []
    # alt. multi-doc summary using 2 most relevant sentences of each per-doc summary
    for ds in doc_summaries:
        best_sents = sorted(ds, key = lambda sent: sent[1], reverse = True)

        for i in range(2):
            sent = best_sents[i]
            print("    {} | [{}]".format(sent[0], sent[1]))
            overall_summary_alt.append(sent[0] + '.')

        print()

    overall_summary_alt_text = ' '.join(overall_summary_alt)

    # TextRank summaries for comparison
    # overall_text = ' '.join(overall_sentences)
    # tr_overall_summary = summarize(overall_text)

    # --- ROUGE Score Evaluation ---
    print("Computing per-document ROUGE scores...", end='\r')
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = []

    overall_text = []
    for i in range(len(targets)):
        print("Computing per-document ROUGE scores... ", i, "/", len(targets), end="\r")

        doc_bow = targets[i][0]
        doc_sentences = targets[i][1]

        doc_raw = ""
        for sentence in doc_sentences:
            doc_raw = doc_raw + sentence[1]

        # textrank summary for comparison
        doc_text = '. '.join([sentence[1] for sentence in doc_sentences])
        overall_text.append(doc_text)
        tr_doc_summary = summarize(doc_text, ratio = 0.005)

        summary_raw = ""
        for sentence in doc_summaries[i]:
            summary_raw = summary_raw + sentence[0]

        sc = rouge.score(summary_raw, tr_doc_summary)
        scores.append(sc)
    print("Computing per-document ROUGE scores... Complete           ")

    overall_text = '. '.join(overall_text)
    
    if len(targets) <= 30:
        print("Generating TextRank overall comparison summary...", end = '\r')
        tr_overall_summary = summarize(overall_text, ratio = 0.005)
        print("Generating TextRank comparison summary... Complete           ")

        print("Computing overall summary ROUGE scores (aggregate and alternative)...", end = '\r')
        overall_rouge_sc = rouge.score(overall_summary_text, tr_overall_summary)
        overall_alt_rouge_sc = rouge.score(overall_summary_alt_text, tr_overall_summary)
        print("Computing overall summary ROUGE scores... Complete                   ")

        if VERBOS:
            print("ROUGE Scores:")
            for i, score in enumerate(scores):
                print("Target", i, ":  ROUGE-1 =", score['rouge1'].precision, " ROUGE-L =", score['rougeL'].precision)

            print("Overall  : ROUGE-1 = {} | ROUGE-L = {}".format(overall_rouge_sc["rouge1"].precision, overall_rouge_sc["rougeL"].precision))
            print("Alt      : ROUGE-1 = {} | ROUGE-L = {}".format(overall_alt_rouge_sc["rouge1"].precision, overall_alt_rouge_sc["rougeL"].precision))


    prec_avg_rouge1, prec_avg_rougeL, rec_avg_rouge1, rec_avg_rougeL = 0, 0, 0, 0

    for score in scores:
        prec_avg_rouge1 += score["rouge1"].precision
        prec_avg_rougeL += score["rougeL"].precision
        rec_avg_rouge1 += score["rouge1"].recall
        rec_avg_rougeL += score["rougeL"].recall

    avgs = [prec_avg_rouge1, prec_avg_rougeL, rec_avg_rouge1, rec_avg_rougeL]

    for i, avg in enumerate(avgs):
        avgs[i] = avg / len(scores)

    if VERBOS:
        print("\nAverage ROUGE score across all documents:")
        print("(Precision)  ROUGE-1 = {} | ROUGE-L = {}\n(Recall)  ROUGE-1 = {} | ROUGE-L = {}".format(*avgs))

    os.makedirs("../plots/", exist_ok = True)

    with open("../plots/run_logs.csv", 'a', encoding = "utf-8") as file:
        file.write("{},{},{},{},{}\n".format(args.num_topics, *avgs))

    time_elapsed = time.time() - start_time

    print("Total time elapsed: {:.2f}s".format(time_elapsed))
