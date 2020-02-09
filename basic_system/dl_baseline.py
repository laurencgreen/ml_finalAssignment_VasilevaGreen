from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Embedding, SpatialDropout1D, LSTM
from keras.callbacks import EarlyStopping
import pandas as pd
import sys
from utils import pandas_explode_column, read_subsets, split_intensity_labels, visualise_distribution
from preprocessing import prep_dl_data

from LSTM import build_lstm_model
from biLSTM import build_bilstm_model
from GRU import build_gru_model


def pad_features(dataframe, tokenizer, MAX_SEQUENCE_LENGTH):
    X = tokenizer.texts_to_sequences(dataframe['tweet'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    return X


def get_features(train_df, validation_df, test_df, tokenizer, MAX_SEQUENCE_LENGTH):
    X_train = pad_features(train_df, tokenizer, MAX_SEQUENCE_LENGTH)
    X_val = pad_features(validation_df, tokenizer, MAX_SEQUENCE_LENGTH)
    X_test = pad_features(test_df, tokenizer, MAX_SEQUENCE_LENGTH)
    print(f"Shape of X_train: {X_train.shape}\n Shape of X_val: {X_val.shape}\n Shape of X_test: {X_test.shape}")
    return X_train, X_val, X_test


def get_labels(train_df, validation_df, test_df):  
    y_train = train_df['intensity_scores'].values.tolist()
    y_val = validation_df['intensity_scores'].values.tolist()
    y_test = test_df['intensity_scores'].values.tolist()
    
    y_train = to_categorical(y_train, num_classes=7)
    y_val = to_categorical(y_val, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)
    print(f"Shape of y_train: {y_train.shape}\n Shape of y_val: {y_val.shape}\n Shape of y_test: {y_test.shape}")
    return y_train, y_val, y_test


def launch_tokenizer(dataframe, token_column_name, numwords):
    tokenizer = Tokenizer(num_words=numwords, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)                      
    tokenizer.fit_on_texts(dataframe[token_column_name].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    return tokenizer, word_index


def define_embedding_dimensions(EMBEDDINGS):
    """Input embeddings as parameter. Since SSWE have embedding dimensions of 50 whilst fasttext and glove
    have 300; Return embedding dimension as integer"""
    if "sswe" in EMBEDDINGS:
        EMBEDDING_DIM = 50  # sswe have dims==50 
    else:
        EMBEDDING_DIM = 300  # fasttext/glove dims==300 
    return EMBEDDING_DIM


def dl_main(train_path, validation_path, test_path, model_name, PREPROCESSING="False", EMBEDDINGS="None"):
    # SET FIXED PARAMTERS IN CAPS
    MAX_SEQUENCE_LENGTH = 250  # Max number of words in each text.
    EPOCHS = 1 #20
    BATCH_SIZE = 64 #64
    EMBEDDING_DIM = define_embedding_dimensions(EMBEDDINGS)
    
    if PREPROCESSING == "False":
        MAX_NB_WORDS = 5000 #The maximum number of words to be used. (most frequent)

        print('Loading raw data...')
        columns = ["ID", "tweet", "affect_dimension", "intensity_class"]
        train_df, validation_df, test_df = read_subsets(train_path, validation_path, test_path, columns)
        train_df, validation_df, test_df = split_intensity_labels([train_df, validation_df, test_df])
    else:
        MAX_NB_WORDS = 4608  # match shape of preprocessed data

        print('Loading preprocessed data...')
        columns = ["ID", "affect_dimension", "intensity_scores", "intensity_descriptions", "tweet"]
        train_df, validation_df, test_df = read_subsets(train_path, validation_path, test_path, columns)

    # SENTENCES = train_df["tweet"].values.tolist()

    visualise_distribution([train_df, validation_df, test_df], "data/visualisations/")
       
    tokenizer, word_index = launch_tokenizer(train_df, "tweet", MAX_NB_WORDS)  # fitting tokenizer on train set
  
    print("Transforming train, validation and test to sequences and padding...")
    X_train, X_val, X_test = get_features(train_df, validation_df, test_df, tokenizer, MAX_SEQUENCE_LENGTH)
    y_train, y_val, y_test = get_labels(train_df, validation_df, test_df)
    
    print("Building the model...")
    if model_name == "lstm":
        model = build_lstm_model(MAX_NB_WORDS, EMBEDDING_DIM, word_index, EMBEDDINGS)
    elif model_name == "bilstm":
        model = build_bilstm_model(MAX_NB_WORDS, EMBEDDING_DIM, word_index, EMBEDDINGS)
    elif model_name == "gru":
        model = build_gru_model(MAX_NB_WORDS, EMBEDDING_DIM, word_index, EMBEDDINGS)
    else:
        print("Default LSTM model")
        model = build_lstm_model(MAX_NB_WORDS, EMBEDDING_DIM, word_index, EMBEDDINGS)
    
    print("Fitting the model...")
    history = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    return y_test, X_test, model, history
    
if __name__ == "__main__":
    dl_main(train_path, validation_path, test_path, model_name, PREPROCESSING="False", EMBEDDINGS="None")

