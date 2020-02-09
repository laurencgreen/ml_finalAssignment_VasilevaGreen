import pandas as pd
pd.set_option('display.max_columns', 500)
import spacy
nlp = spacy.load('en_core_web_sm')
import ast
import re
import string
from bs4 import BeautifulSoup
from html import unescape
from tqdm import tqdm
from wordsegment import load, segment
from utils import read_csv, save_csv, pandas_explode_column, pandas_column_list_drop


def replace_abbreviations(tweet_string):
    s = open("utils/abbreviations_dict.txt", "r").read()
    negation_dict = ast.literal_eval(s)
    lower_tweet = tweet_string.lower()
    for abbrev, full in negation_dict.items():
        lower_tweet = lower_tweet.replace("’", "'")
        lower_tweet = lower_tweet.replace(abbrev.lower(), full)
    return lower_tweet

def spacy_tokenization(tweet_string):
    """Input string, apply spacy and tokenize. Return as a list of spacy tokens"""
    spacy_string = nlp(tweet_string.lower())
    return spacy_string

def remove_whitespace_entities(spacy_token_list):
    "Input spacy token list, remove extra whitespace. Return as list of filtered spacy tokens "
    filtered_content = []
    for token in spacy_token_list:
        if not token.text.isspace() or token.text != ' ':
            if token.text != '️':
                filtered_content.append(token.text)
    return filtered_content

def strip_all_ascii(content):
    clean_content = BeautifulSoup(unescape(content), "lxml").text
    additional_chars = ["\n", "\\n"]
    for i in additional_chars:
        clean_content = clean_content.replace(i, "")
    return clean_content


def replace_numbers(content):
    """Input string, remove all numbers"""
    regex_numbers = re.compile(r"\b\d+\b")
    content_number = regex_numbers.sub(" number ", content)
    return content_number


def hashtags(tweet):
    """Input tweet string. Split ONLY words which are inside/involved a twitter hashtag. Replace "#" with 'HASHTAG'
    string as indicative label. Return as filtered string.
    E.g.
    Input = "I like eating cupcakes #theytastesonice" #-> only words inside hashtag will be split to avoid: "cup", "cake"
    Output = "I like eating cupcakes HASHTAG they taste so nice"
     """
    tweet = tweet.split()
    load()
    for i, token in enumerate(tweet):
        if token.startswith("#"): #only want to split hashtag words (NOT others)
            tweet[i] = "# " + " ".join(segment(token))
    filtered_tweet = " ".join(tweet)
    filtered_tweet = filtered_tweet.replace("#", " hashtag ") # HASHTAG as indicative label
    return filtered_tweet


def user_mentions(content):
    regex_mentions = re.compile(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)")
    tweet_mention = regex_mentions.sub(" user_mention ", content)
    return tweet_mention


def tweet_specific(tweet):
    """replace: USERS (mentions), hashtags, urls, emails, percents, money, 
    time, dates, phone numbers, numbers, emoticons ??????????????????????????"""
    content_mention = user_mentions(tweet)
    content_hashtags = hashtags(content_mention)
    return content_hashtags


def remove_punctuation(content):
    """Input list fo spacy tokens. Strip all punctuation(EXCEPT '?', '!' and emoticons). Return as string"""
    punctuation = string.punctuation + "’" + "…" + "»" + "«" + '“' + '”' + "..." + "" + "©" + "" + "" + "•" + \
                  "–" + "‘" + "—" + "®" + "ï" + "¿" + "." # extra punct to remove
    punctuation = punctuation.replace("?", "")
    punctuation = punctuation.replace("!", "")
    punct_free_tweet = []
    for token in content:
        if token.text not in punctuation:
            punct_free_tweet.append(token)
    return punct_free_tweet


def upper_markers(tweet_string):
    """Input tweet as string. Insert "xxall " prior to token if entire token is capitalised
    and "xxstr " prior to token if token starts with capital. Return as preprocessed string"""
    tokens = tweet_string.split()
    for i, token in enumerate(tokens):
        if token.isupper() == True:  # whole token is upper
            all_upper = "xxupp_all " + token
            tokens[i] = all_upper
        elif token[0].isupper():  # token starts with upper
            beg_upper = "xxupp_str " + token
            tokens[i] = beg_upper
        else:
            pass
    upperall_string = " ".join(tokens)
    return upperall_string
    

def preprocess_tweets(df, column_header, uppercase=False, abbrev=False, strip_ascii=False, 
twitter_specific=False, number_replace=False, tokenize=False, punct=False,
 whitespace=False):

    df, tweet_list = pandas_column_list_drop(df, column_header)

    final_list = []
    for tweet in tqdm(tweet_list):
        if uppercase==True:
            tweet = upper_markers(tweet)  # insert indicative token for capitalisation (whole token / token[0])
        if abbrev==True:
            tweet = replace_abbreviations(tweet)
        if strip_ascii==True:
            tweet = strip_all_ascii(tweet)
        if twitter_specific==True:
            tweet = tweet_specific(tweet)  # clean with tweet specifics (mentions, # etc)
        if number_replace==True:
            tweet = replace_numbers(tweet) # replace all numbers with
        if tokenize==True: 
            tweet = spacy_tokenization(tweet)  # apply spacy for tokenization, convert to lowercase
        if punct==True:
            tweet = remove_punctuation(tweet)
        if whitespace==True:
            tweet = remove_whitespace_entities(tweet)  # remove whitespace
        print(tweet)
        final_list.append(tweet)
    return final_list

def prep_dl_data(dataframe):
    preprocessed_tweet_list = preprocess_tweets(dataframe, "tweet",
    uppercase=True, 
    twitter_specific=True, 
    number_replace=True)
    return preprocessed_tweet_list

def prep_ml_data(dataframe):
    preprocessed_tweet_list = preprocess_tweets(dataframe, "tweet",
    uppercase=True, 
    abbrev=True, 
    strip_ascii=True, 
    twitter_specific=True, 
    number_replace=True, 
    tokenize=True, 
    punct=True,
    whitespace=True)
    return preprocessed_tweet_list

def preprocessing_main(filepath, model_name, subset_name):
    columns = ["ID", "tweet", "affect_dimension", "intensity_class"]
    df = read_csv(filepath, columns=columns)
    
    df = pandas_explode_column(df, "intensity_class", "intensity_scores", "intensity_descriptions", delimiter=":")
    
    if model_name == "dl":
        preprocessed_tweet_list = prep_dl_data(df)
        preprocessed_filename = "dl_" + subset_name
        preprocessed_filepath = "data/preprocessed_data/DL_data"
    
    elif model_name == "ml":
        preprocessed_tweet_list = prep_ml_data(df)
        preprocessed_tweet_list = [' '.join(x) for x in preprocessed_tweet_list]  # remove outer list
        preprocessed_filename = "ml_" + subset_name
        preprocessed_filepath = "data/preprocessed_data/ML_data"
    
    else:
        print("No Preprocessing for models other than dl and ml")


    df["tweets"] = preprocessed_tweet_list
    save_csv(df, preprocessed_filepath, preprocessed_filename, "\t")
    
    print("Finished Preprocessing Programme")
