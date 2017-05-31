import pandas as pd
import numpy as np
import nltk
from collections import Counter
from nltk.corpus import stopwords
from sklearn.metrics import log_loss
from scipy.optimize import minimize
stops = set(stopwords.words("english"))
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import multiprocessing
import difflib

'''
Note: Python 3.6 version is recommended

Based on kernel: Matching ¿Que? for Quora - End to End 0.33719 PB
Actual score: 0.34153
'''


train = pd.read_csv('input/train.csv')[:]
test = pd.read_csv('input/test.csv')[:]

def diff_ratios(st1, st2):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(st1).lower(), str(st2).lower())
    return seq.ratio()

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


def get_weight(count, eps=500, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)


train_qs = pd.Series(train['question1'].tolist() + train['question2'].tolist()).astype(str)
test_qs = pd.Series(test['question1'].tolist() + test['question2'].tolist()).astype(str)
words = (" ".join(train_qs)).lower().split() + (" ".join(test_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}


def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def get_unigrams(que):
    return [word for word in nltk.word_tokenize(que.lower()) if word not in stops]

def get_common_unigrams(row):
    return len( set(row["unigrams_ques1"]).intersection(set(row["unigrams_ques2"])) )

def get_common_unigram_ratio(row):
    return float(row["zunigrams_common_count"]) / max(len( set(row["unigrams_ques1"]).union(set(row["unigrams_ques2"])) ),1)

def get_bigrams(que):
    return [i for i in nltk.ngrams(que, 2)]

def get_common_bigrams(row):
    return len( set(row["bigrams_ques1"]).intersection(set(row["bigrams_ques2"])) )

def get_common_bigram_ratio(row):
    return float(row["zbigrams_common_count"]) / max(len( set(row["bigrams_ques1"]).union(set(row["bigrams_ques2"])) ),1)

train['question1_nouns'] = train.question1.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
train['question2_nouns'] = train.question2.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])

train['z_len1'] = train.question1.map(lambda x: len(str(x)))
train['z_len2'] = train.question2.map(lambda x: len(str(x)))
train['z_word_len1'] = train.question1.map(lambda x: len(str(x).split()))
train['z_word_len2'] = train.question2.map(lambda x: len(str(x).split()))
train['z_noun_match'] = train.apply(lambda r: sum([1 for w in r.question1_nouns if w in r.question2_nouns]), axis=1)

train['z_match_ratio'] = train.apply(lambda r: diff_ratios(r.question1, r.question2), axis=1)

train['z_word_match'] = train.apply(word_match_share, axis=1, raw=True)
train['z_tfidf_word_match'] = train.apply(tfidf_word_match_share, axis=1, raw=True)

train["unigrams_ques1"] = train['question1'].apply(lambda x: get_unigrams(str(x)))
train["unigrams_ques2"] = train['question2'].apply(lambda x: get_unigrams(str(x)))
train["zunigrams_common_count"] = train.apply(lambda r: get_common_unigrams(r),axis=1)
train["zunigrams_common_ratio"] = train.apply(lambda r: get_common_unigram_ratio(r), axis=1)
train["bigrams_ques1"] = train["unigrams_ques1"].apply(lambda x: get_bigrams(x))
train["bigrams_ques2"] = train["unigrams_ques2"].apply(lambda x: get_bigrams(x))
train["zbigrams_common_count"] = train.apply(lambda r: get_common_bigrams(r),axis=1)
train["zbigrams_common_ratio"] = train.apply(lambda r: get_common_bigram_ratio(r), axis=1)

#-------------------------------------------------------------------------------------------------

test['question1_nouns'] = test.question1.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
test['question2_nouns'] = test.question2.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])

test['z_len1'] = test.question1.map(lambda x: len(str(x)))
test['z_len2'] = test.question2.map(lambda x: len(str(x)))
test['z_word_len1'] = test.question1.map(lambda x: len(str(x).split()))
test['z_word_len2'] = test.question2.map(lambda x: len(str(x).split()))
test['z_noun_match'] = test.apply(lambda r: sum([1 for w in r.question1_nouns if w in r.question2_nouns]), axis=1)

test['z_match_ratio'] = test.apply(lambda r: diff_ratios(r.question1, r.question2), axis=1)

test['z_word_match'] = test.apply(word_match_share, axis=1, raw=True)
test['z_tfidf_word_match'] = test.apply(tfidf_word_match_share, axis=1, raw=True)

test["unigrams_ques1"] = test['question1'].apply(lambda x: get_unigrams(str(x)))
test["unigrams_ques2"] = test['question2'].apply(lambda x: get_unigrams(str(x)))
test["zunigrams_common_count"] = test.apply(lambda r: get_common_unigrams(r),axis=1)
test["zunigrams_common_ratio"] = test.apply(lambda r: get_common_unigram_ratio(r), axis=1)
test["bigrams_ques1"] = test["unigrams_ques1"].apply(lambda x: get_bigrams(x))
test["bigrams_ques2"] = test["unigrams_ques2"].apply(lambda x: get_bigrams(x))
test["zbigrams_common_count"] = test.apply(lambda r: get_common_bigrams(r),axis=1)
test["zbigrams_common_ratio"] = test.apply(lambda r: get_common_bigram_ratio(r), axis=1)

train = train.fillna(-1)
test = test.fillna(-1)

#train.to_csv('train.csv', index=False)
#test.to_csv('test.csv', index=False)

col = [c for c in train.columns if c[:1]=='z']

pos_train = train[train['is_duplicate'] == 1]
neg_train = train[train['is_duplicate'] == 0]
p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
train = pd.concat([pos_train, neg_train])

x_train, x_valid, y_train, y_valid = train_test_split(train[col], train['is_duplicate'], test_size=0.2, random_state=0)

params = {}
params["objective"] = "binary:logistic"
params['eval_metric'] = 'logloss'
params["eta"] = 0.02
params["subsample"] = 0.7
params["min_child_weight"] = 1
params["colsample_bytree"] = 0.7
params["max_depth"] = 4
params["silent"] = 1
params["seed"] = 1632

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
bst = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=50, verbose_eval=100) #change to 5000

d_test = xgb.DMatrix(test[col])
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = test['test_id']
sub['is_duplicate'] = p_test

#df['is_duplicate'] = df['is_duplicate'].map(lambda x: 0.000000000001 if x < 0.0001 else x)
#df['is_duplicate'] = df['is_duplicate'].map(lambda x: 0.999999999999 if x > 0.98 else x)

sub.to_csv('z05_submission_xgb_03.csv', index=False)
print(log_loss(train.is_duplicate, bst.predict(xgb.DMatrix(train[col]))))