import numpy as np
import pandas as pd
from sklearn import svm
import pickle
from utils import create_new_folder, get_column_list, concat_dataframes, split_intensity_labels, read_subsets
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
import sys
sys.path.insert(0, "feature_extraction")
from w2v import open_embeddings_model, embed_sentences
from lex2v import lex_main
# from s2v import dl_sent_matrix

def train_model(train_df, test_df, model_name, EMBEDDINGS, LEXICON):
    """Input X_train, y_train, X_test, y_test and classification model. Print classification metrics -> see
    get_metrics() function"""
    print("Training model")

    model = define_model(model_name)

    if EMBEDDINGS == "False" and LEXICON == "False":
        X_train, y_train, X_test, y_test = encode_bow_data(train_df, test_df)
        clf = bow_pipeline(model)

    else:
        y_train = train_df.intensity_scores.values[:20]
        y_test = test_df.intensity_scores.values[:8]
        train_features_df, test_features_df = get_features(train_df, test_df, EMBEDDINGS, LEXICON)
        X_train = train_features_df.to_dict(orient="records")
        print(X_train)
        X_test = test_features_df.to_dict(orient="records")
        clf = dict_pipeline(model)
    
    clf.fit(X_train, y_train)
    save_classification_model(model_name, clf)
    return X_test, y_test

def define_model(model_name):
    if model_name == "svm":
        model = svm.LinearSVC(max_iter=10000, dual=False, C=0.1)
    return model

def encode_bow_data(train_df, test_df):
    X_train = train_df.tweets.values
    y_train = train_df.intensity_scores.values

    X_test = test_df.tweets.values
    y_test = test_df.intensity_scores.values
    return X_train, y_train, X_test, y_test

def bow_pipeline(model):
    clf = Pipeline([('vect', CountVectorizer()), ('frm', TfidfTransformer()), ('clf', model)])
    return clf

def dict_pipeline(model):
    clf = Pipeline([('vect', DictVectorizer()), ('scaler', StandardScaler(with_mean=False)), ('clf', model)])
    return clf

def save_classification_model(filename, model):
    """Input filename as string and classification model. If not already existing, create following directory and
    save model in given folder: 'basic_system/classification_models/'"""
    cm_folder_path = "basic_system/classification_models/" #classification models to be saved here
    create_new_folder(cm_folder_path)
    with open(cm_folder_path + filename, "wb") as picklefile:
        pickle.dump(model, picklefile)
    print(f"Saved {filename} model in {cm_folder_path}")


def prepare_data(df, EMBEDDINGS):
    if "s_" in EMBEDDINGS:
        tweet_list = df["tweets"].values.tolist() 

    else:
        tweet_list = [tweet.split() for tweet in df["tweets"].tolist()]  # list of list of tokens
        df = pd.DataFrame({0: tweet_list})
        df.columns= [["tweets"]]
    return tweet_list, df


def get_features(train_df, test_df, EMBEDDINGS, LEXICON):
    train_df = train_df.iloc[:20]
    test_df = test_df.iloc[:8]
    train_tweet_list, train_df = prepare_data(train_df, EMBEDDINGS)
    test_tweet_list, test_df = prepare_data(test_df, EMBEDDINGS)

    if EMBEDDINGS != "False":
        if EMBEDDINGS == "s_infersent":
            print("Lisas code here")
        elif "s_sentT" in EMBEDDINGS:
            embedding_model = open_model(EMBEDDINGS)
            train_df = sentence_transformer(train_tweet_list, embedding_model)
            test_df = sentence_transformer(test_tweet_list, embedding_model)
        else:
            embeddings_index = open_embeddings_model(EMBEDDINGS)
            train_df = embed_sentences(train_df, "tweets", embeddings_index, EMBEDDINGS)

            test_df = embed_sentences(test_df, "tweets", embeddings_index, EMBEDDINGS)
    
    if LEXICON != "False":
        train_lex_list = lex_main(train_tweet_list, LEXICON)
        test_lex_list = lex_main(test_tweet_list, LEXICON)
        train_df["lex"] = train_lex_list
        test_df["lex"] = test_lex_list

    train_df = train_df.astype(str)
    test_df = test_df.astype(str)
    return train_df, test_df


def encode_data(train_df, validation_df, test_df, EMBEDDINGS, LEXICON):
    y_train, y_test = get_labels(train_df, test_df, "intensity_scores")
    train_features_df, test_features_df = get_features(train_df, test_df, EMBEDDINGS, LEXICON)
    # X_train, X_val, X_test = fit_transform_vec(train_features_df, validation_features_df, test_features_df)
    return X_train, y_train, X_val, y_val, X_test, y_test


def ml_main(train_filepath, validation_filepath, test_filepath, model_name, PREPROCESSING="False", EMBEDDINGS="False", LEXICON="False"):

    if PREPROCESSING == "False":
        print('Loading raw data...')
        columns = ["ID", "tweets", "affect_dimension", "intensity_class"]
        train_df, validation_df, test_df = read_subsets(train_filepath, validation_filepath, test_filepath, columns)
        train_df, validation_df, test_df = split_intensity_labels([train_df, validation_df, test_df])

    else:
        print('Loading preprocessed data...')
        columns = ["ID", "affect_dimension", "intensity_scores", "intensity_descriptions", "tweets"]
        train_df, validation_df, test_df = read_subsets(train_filepath, validation_filepath, test_filepath, columns)

    train_df = pd.concat([train_df, validation_df], axis=0)

    # X_train, y_train, X_val, y_val, X_test, y_test = encode_data(train_df, validation_df, test_df, EMBEDDINGS, LEXICON)
    # print(X_train.shape, X_val.shape, X_test.shape)

    X_test, y_test = train_model(train_df, test_df, model_name, EMBEDDINGS, LEXICON)

    print("Finished training and testing")
    return X_test, y_test

if __name__ == "__main__":
    ml_main(train_filepath, test_filepath)
