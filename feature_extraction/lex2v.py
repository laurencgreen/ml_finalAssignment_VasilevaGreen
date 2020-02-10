import pandas as pd
from utils import average_scores

def lexobj_nrc():
    lex_list_dict = []
    lex_name = "data/lexicons/NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt"
    df = pd.read_csv(lex_name, sep = "\t")
    df.columns = ["Word", "Valence", "Arousal", "Dominance"]
    for i, row in df.iterrows():
        Dlex = {}
        entry = df.loc[i]["Word"]
        score = df.loc[i]["Valence"]
        Dlex[entry] = score
        lex_list_dict.append(Dlex)
    return lex_list_dict


def apply_lexicon(data, lex_list_dict):
    processed = []
    for i, tw in enumerate(data):
        print(f"{i}/ {len(data)}")
        processed_tweet = []
        for token in tw:
            for d in lex_list_dict:
                for entry, score in d.items():
                    if str(token) == str(entry):
                        processed_tweet.append(score)
                    else:
                        processed_tweet.append(0)  # token is not sentiment specific
        processed.append(processed_tweet)       
    averages = average_scores(processed)
    print(len(averages)) 
    return averages

def lex_main(data, LEXICON_TYPE="nrc"): #parameter lexicon type, # data
    # LEXICON_TYPE = "scl"

    # if LEXICON_TYPE == "scl":
    #     lexicon_dict = lexobj_scl()
    # elif LEXICON_TYPE == "nrc":
    #     lexicon_dict = lexobj_nrc()
    # elif LEXICON_TYPE == "combo":
    #     lexicon_dict = lexobj_combined()
    # else:
    print("Default NRC Lexicon utilised")  # choose whatever here ?
    lexicon_dict = lexobj_nrc()

    avg_tweet_scores = apply_lexicon(data, lexicon_dict)
    print("Lexicon Done")
    return avg_tweet_scores
