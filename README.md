# PearNet-v_oc-

Report: https://www.overleaf.com/read/bnrzjqndtvng

## 1) Download train, validation and test folders
*https://competitions.codalab.org/competitions/17751#learn_the_details-datasets* 
#### Save each of the .txt files into the directory: "data/"
* "2018-Valence-oc-En-train.txt"    
* "2018-Valence-oc-En-dev.txt"       
* "2018-Valence-oc-En-test-gold.txt"

## 2) Download following word embedding files:
#### Save into the directory: "data/embeddings/"
* "glove.840B.300d.txt" -> https://www.floydhub.com/redeipirati/datasets/glove-840b-300d
* "wiki-news-300d-1M.vec" -> https://fasttext.cc/docs/en/english-vectors.html (only save vec file itself into directory)
* "sswe-u.txt", "sswe-h.txt", "sswe-r.txt" -> http://ir.hit.edu.cn/~dytang/paper/sswe/embedding-results.zip

## 3) Run main.py using the arguments below:

| Argument[i] (argv)       | Label           |Input           | 
| ------------- |------------- |:-------------:| 
| argv[1] | MODEL_NAME           |"svm" / "lstm"                   |  
| argv[2] | EMBEDDINGS           |"fasttext" / "glove" / "sswe-u" / "sswe-h" / "sswe-r" / s_infersent / s_sentT_bert-base-nli-mean" / "False"    |  
| argv[3] | PREPROCESSING           |"True" vs "False| 
| argv[4] | LEXICON           |"True" (default NRC) vs "False| 


e.g. *python main.py lstm sswe-u True False*

word embedding models = {1: fasttext, 2: glove, 3-5: three different SSWE}
sentence embedding models = {1: sentenceTransformerBERT-Base-Nli-Mean, 2: Infersent}
* so far runs:
 * LSTM + all five word embedding models and two sentence embedding models 
 * SVM + all five word embedding models and two sentence embedding model (bert and infersent) + NRC lexicon as features
each of the models and combinations can be run both with and without preprocessing

---

ml vs ensemble methods
- baseline: svm
- lstm ----> bi-lstm???
- word embeddings
-- sswe
-- ft / other cbow?
- lexicons
- lstm + deepmoji
- predictive modelling


problems with word_segment (i.e. hashtag splitting/tokenizing in some cases):
- segment ("time" -> "ti", "ame" / "wimbeldon" -> "wi", "ambeldon" / "climb" -> "cli", "amb") -> keeps adding a...


line 32: dev: "'We can throw stones, complain about them, stumble on them, climb over them, or build with them.' — William Arthur Ward. #inspiring" -> "throw stones complain stumble cli amb build william arthur ward hashtag inspiring"
---> indicate quote?
