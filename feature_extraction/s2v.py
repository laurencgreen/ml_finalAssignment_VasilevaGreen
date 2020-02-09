from sentence_transformers import SentenceTransformer
# https://pypi.org/project/sentence-transformers/
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys

sys.path.insert(0, "feature_extraction/InferSent-master")
from models import InferSent
# #https://github.com/facebookresearch/InferSent

import torch
import nltk
nltk.download('punkt')


def create_dl_sent_embedding_matrix(MAX_NB_WORDS, EMBEDDING_DIM, word_index, EMBEDDINGS):
    """Load specified sentence embeddings. Create and return embedding matrix for weight
    paramter in DL model"""
    model = open_model(EMBEDDINGS)
    print(model.shape)

    # sentence_embeddings = sentence_transformer(sentence_list, model)

    ##### need to adapt this still for dl model
    # print("Creating embedding matrix")
    # embeddings_index = open_embeddings_model(EMBEDDINGS)



def open_model(transformer_name):
    print(f"Opening {transformer_name} sentence embeddings")
    if "bert-base-nli-mean" in transformer_name:
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        
    elif "s_infersent" in transformer_name:
        V = 2
        MODEL_PATH = 'encoder/infersent%s.pkl' % V
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
        model = InferSent(params_model)
        model.load_state_dict(torch.load(MODEL_PATH, encoding="utf-8"))

        # setting word vector path for the model
        W2V_PATH = 'data/embeddings/crawl-300d-2M.vec' #need to adapt the path
        model.set_w2v_path(W2V_PATH)

    return model

def sentence_transformer(sentence_list, model):
    sentence_embeddings = model.encode(sentence_list)
    embeddings = []
    for sentence, embedding in tqdm(zip(sentence_list, sentence_embeddings)):
        print(len(embedding)) # dimensions 768
        embeddings.append(embedding)
        df = pd.DataFrame.from_records(embeddings)
    return sentence_embeddings
