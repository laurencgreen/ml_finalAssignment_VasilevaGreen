import numpy as np
import pandas as pd
from utils import read_csv, save_csv, slice_list, average_scores
from itertools import chain 
from collections import Counter
import sys
import codecs
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from pathlib import Path
from tqdm import tqdm 

def create_sswe_matrix(embeddings_index, df, column_name):
    tweet_list = df[column_name].values.tolist()
    tweet_list  = [val for sublist in tweet_list for val in sublist]
    vocab = build_vocab(tweet_list)
    embeddings_index = load_senti_emb(embeddings_index, vocab)
    sswe_matrix = create_ml_embedding_matrix(df, column_name, embeddings_index)
    return sswe_matrix


def create_ml_embedding_matrix(df, column_name, embeddings_index):
    """Load specified word embeddings. Create and return as list of lists of word vectors for each tweet"""
    print("Creating embedding matrix")
    tweets = df[column_name].values.tolist()
    tweets = [val for sublist in tweets for val in sublist]
    embedding_arrays = []
    for sent in tweets:   
        sent_arrays = []
        for word in sent:  
            if word not in embeddings_index:
                sent_arrays.append(embeddings_index["unknown"])
            else:
                sent_arrays.append(embeddings_index[word])
        embedding_arrays.append(sent_arrays)
    s_embeddings = [] # convert from list of arrays to list of lists
    for sent_embed in embedding_arrays:
        w_embeddings = []
        for w_embed in sent_embed:
            w_embeddings.append(w_embed.tolist())
        s_embeddings.append(w_embeddings)
    return s_embeddings


def create_dl_word_embedding_matrix(MAX_NB_WORDS, EMBEDDING_DIM, word_index, EMBEDDINGS):
    """Load specified word embeddings. Create and return embedding matrix for weight
    paramter in DL model"""
    print("Creating embedding matrix")
    embeddings_index = open_embeddings_model(EMBEDDINGS)
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0: # words not in embedding_index=0
            embedding_matrix[i] = embedding_vector
        else:
            pass
    return embedding_matrix

def open_embeddings_model(EMBEDDINGS):
    if EMBEDDINGS == "glove":
        embeddings_index = glove_embeddings()
    elif EMBEDDINGS == "fasttext":
        embeddings_index = fasttext_embeddings()
    elif "sswe" in EMBEDDINGS:
        embeddings_index = sswe_embeddings(EMBEDDINGS)
    else:
        print("Default fasttext embeddings")
        embeddings_index = fasttext_embeddings()
    return embeddings_index


def glove_embeddings():
    """Load GloVe word embeddings. Return as embeddings index dictionary"""
    path = 'data/embeddings/glove.twitter.27B.100d.txt'
    w2vfile = 'data/embeddings/glove.twitter.27B.100d.w2v.txt'
    if not Path(w2vfile).is_file():
        glove2word2vec(path, w2vfile)
    embeddings_index = KeyedVectors.load_word2vec_format(w2vfile, binary=False)
    return embeddings_index


def fasttext_embeddings():
    """Load FastText word embeddings. Return as embeddings index dictionary"""
    filepath = "data/embeddings/wiki-news-300d-1M.vec"
    embeddings_index = dict()
    f = codecs.open(filepath, encoding= "utf-8")
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def sswe_embeddings(EMBEDDINGS):
    embeddings_index = {}
    with open("data/embeddings/" + EMBEDDINGS + ".txt", "r",
     encoding="utf-8") as f:
        for line in f.readlines():
            values = line.split()
            word = line[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    return embeddings_index

def load_senti_emb(embeddings_index, vocab):
    EMBEDDINGS = {}
    num_oov = 0
    word_in_sswe = 0
    for word in vocab:
        if word in embeddings_index:
            word_in_sswe += 1
            vec = embeddings_index[word]
            EMBEDDINGS[word] = (vec - min(vec)) / np.add(max(vec), -min(vec)) * 2 - 1
        else:
            num_oov += 1
            EMBEDDINGS[word] = np.random.uniform(-1, 1, 300)
    print('Word in SSWE is', word_in_sswe, 'Total num of oov is', num_oov)
    return EMBEDDINGS


def embed_sentences(df, column_name, embeddings_index, EMBEDDINGS):
    if "sswe" not in EMBEDDINGS:
        w_embeddings = create_ml_embedding_matrix(df, column_name, embeddings_index)
    else:
        w_embeddings = create_sswe_matrix(embeddings_index, df, column_name)
    averaged_w_embeddings = [average_scores(v) for v in w_embeddings]  # average vectors for each word
    s_embeddings = average_scores(averaged_w_embeddings)  # average word vectors for each sentence
    df[EMBEDDINGS.lower()] = s_embeddings
    return df


def build_vocab(sentences):
    print("building vocab")
    word_counts = Counter(chain(*sentences))
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
    return vocabulary