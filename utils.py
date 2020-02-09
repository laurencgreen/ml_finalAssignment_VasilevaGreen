import pandas as pd
import os
import pickle
from itertools import islice 
import matplotlib.pyplot as plt
import seaborn as sns

def read_csv(filepath, delimiter="\t", index=False, columns=None):
    df = pd.read_csv(filepath, delimiter=delimiter, encoding="utf-8")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')] #drop any columns unnamed
    if index != False:
        df.set_index(index, inplace=True)
    if columns != None:
        df.columns = columns
    return df

def visualise_distribution(dataframe_list, filepath):
    create_new_folder(filepath)
    all_data = pd.concat(dataframe_list)
    fig = plt.figure(figsize=(8,5))
    sns.barplot(x = all_data["intensity_scores"].unique(), 
                y=all_data["intensity_scores"].value_counts())

    plt.title(f"Distribution of Number of Instances per Intensity Score", fontsize=15)
    plt.ylabel("Number of Instances", fontsize=15)
    plt.xlabel("Intensity Labels", fontsize=15)
    plt.show()
    fig.savefig(filepath + "/distribution_skew.png", dpi=fig.dpi)
    print(f"Saved figure in {filepath}")


def save_csv(df, filepath, filename, sep="\t"):
    create_new_folder(filepath)
    if sep == "\t":
        df.to_csv(filepath + "/" + filename + ".tsv", sep=sep)
    else:
        df.to_csv(filepath + "/" + filename + ".csv", sep=sep)
    print(f"Saved {filename} as csv")


def concat_dataframes(list_of_dataframes):
    dataframe_list = pd.concat(list_of_dataframes)
    return dataframe_list

def create_new_folder(path):
    """If directory doesn't exist, create new filepath"""
    if not os.path.exists(path):
        os.makedirs(path)   # create folder if not already existing
        print(f"created {path}")

def pandas_explode_column(df, column_name, column_1, column_2, delimiter=","):
    df[column_1], df[column_2] = df[column_name].str.split(delimiter, 1).str
    df.drop([column_name], axis=1, inplace=True)
    return df

def pandas_column_list_drop(df, column_header):
    """Insert pandas dataframe and column header as string. Covert column to list and drop column from dataframe.
     Return both dataframe and column list"""
    tweet_list = df[column_header].tolist()
    df.drop([column_header], axis=1, inplace=True)
    return df, tweet_list

def open_pickle_model(filename):
    model = pickle.load(open(filename, "rb"))
    return model

def get_column_list(df, column_header):
    """Input dataframe and name of column header. Return single column as a list"""
    column_list = df[column_header].to_list()
    return column_list

def slice_list(flattened_list, list_of_lengths):
    """# split list into list of lists using indices from original
    https://www.geeksforgeeks.org/python-split-a-list-into-sublists-of-given-lengths/"""
    flattened_list = iter(flattened_list) 
    sliced_list_of_lists = [list(islice(flattened_list, elem)) for elem in list_of_lengths] 
    return sliced_list_of_lists

def average_scores(number_list_of_lists):
    """Input list of lists of numbers (ints or floats) - each inner list referring to sentnce"""
    for i, l in enumerate(number_list_of_lists):
        number_list_of_lists[i] = sum(l) / len(l)
    return number_list_of_lists

def read_subsets(train_path, validation_path, test_path, columns):
    train_df = read_csv(train_path, columns=columns)
    validation_df = read_csv(validation_path, columns=columns)
    test_df = read_csv(test_path, columns=columns)
    return train_df, validation_df, test_df

def split_intensity_labels(df_list, delimiter=":"):
    for i, df in enumerate(df_list):
        df_list[i] = pandas_explode_column(df, "intensity_class", "intensity_scores", "intensity_descriptions", delimiter=delimiter)
    train, val, test = df_list
    return train, val, test