import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from utils import open_pickle_model
import os
import sys
from preprocessing import preprocessing_main
sys.path.insert(0, "evaluation")
from evaluation import evaluation_main


sys.path.insert(0, "basic_system")
from basic_system.ml_baseline import ml_main

from basic_system.dl_baseline import dl_main
# pip freeze > requirements.txt

def main(argv=None):

    TRAIN_FILEPATH = "data/2018-Valence-oc-En-train.txt"
    VALIDATION_FILEPATH = "data/2018-Valence-oc-En-dev.txt"
    TEST_FILEPATH = "data/2018-Valence-oc-En-test-gold.txt"
    
    if argv is None:
        argv = sys.argv
        MODEL_NAME = argv[1]  # SVM / LSTM / bilstm / gru
        EMBEDDINGS = argv[2]  # fasttext / glove / False / sswe-r / sswe-u / sswe-h / s_sentT_bert-base-nli-mean / s_infersent
        PREPROCESSING = argv[3]  # True vs False
        LEXICON = argv[4]  # True vs False 

    dl_models = ["lstm", "bilstm", "gru"]  # dl models utilised 

    if MODEL_NAME in dl_models:
        model_type = "dl"
        path = "data/preprocessed_data/DL_data"
    else:
        model_type= "ml"
        path = "data/preprocessed_data/ML_data"
        
    if PREPROCESSING == "True":
        if not os.path.exists(path):  # check if already been preprocessed before 
            print("Preprocessing files")
            preprocessing_main(TRAIN_FILEPATH, model_type, "train")
            preprocessing_main(VALIDATION_FILEPATH, model_type, "validation")
            preprocessing_main(TEST_FILEPATH, model_type, "test")
        else:
            print(f"Preprocessed files in directory: {path}")
        TRAIN_FILEPATH = path + "/" + model_type + "_train.tsv"
        VALIDATION_FILEPATH = path + "/" + model_type + "_validation.tsv"  
        TEST_FILEPATH = path + "/" + model_type + "_test.tsv"

    
    if MODEL_NAME in dl_models:
        print(f"Training {MODEL_NAME} model")
        y_test, X_test, model, history = dl_main(TRAIN_FILEPATH, VALIDATION_FILEPATH,
        TEST_FILEPATH, MODEL_NAME, PREPROCESSING, EMBEDDINGS)

        print("Evaluating model")
        evaluation_main(y_test, X_test, model, MODEL_NAME, history)

    elif MODEL_NAME == "svm":
        print("Training SVM Model")
        X_test, y_test = ml_main(TRAIN_FILEPATH, VALIDATION_FILEPATH, TEST_FILEPATH, MODEL_NAME,
        PREPROCESSING, EMBEDDINGS, LEXICON)
        print("Loading model back in")
        trained_model = open_pickle_model("basic_system/classification_models/" + MODEL_NAME)
        print("Evaluating model")
        evaluation_main(y_test, X_test, trained_model, MODEL_NAME)

    else:
        print("Enter either 'svm', 'lstm' as model names")


    print("Finished Programme")
    
if __name__ == "__main__":
    main()
