from keras.models import Sequential
from keras.layers import Dense, Embedding, SpatialDropout1D, LSTM
import sys

sys.path.insert(0, "feature_extraction")
from w2v import create_dl_word_embedding_matrix
from s2v import create_dl_sent_embedding_matrix


def create_embedding_matrix(MAX_NB_WORDS, EMBEDDING_DIM, word_index, EMBEDDINGS):
    if "s_" in EMBEDDINGS:
        print("sentence embeddings")
        embedding_matrix = create_dl_sent_embedding_matrix(MAX_NB_WORDS, EMBEDDING_DIM, word_index, EMBEDDINGS)
    else:
        embedding_matrix = create_dl_word_embedding_matrix(MAX_NB_WORDS, EMBEDDING_DIM, word_index, EMBEDDINGS)
    return embedding_matrix


def build_lstm_model(MAX_NB_WORDS, EMBEDDING_DIM, word_index, EMBEDDINGS):
    model = Sequential()
    if EMBEDDINGS == "False":
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM))
    else:
        print(f"Loading {EMBEDDINGS} word embeddings")
        word_embedding_matrix = create_embedding_matrix(MAX_NB_WORDS, EMBEDDING_DIM, word_index, EMBEDDINGS)
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, weights=[word_embedding_matrix]))

    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model 


