import numpy as np
import pandas as pd
tr = pd.read_csv('./input/train.csv')
te = pd.read_csv('./input/test.csv')
from nltk.corpus import stopwords

# see: https://www.kaggle.com/dmitriyab/pandas-model-no-ml-lb-0-356
# Gives a score of: 0.35633

'''
First up is word_match_share. This is the one feature in our model. 
The version below is Pandas-centric, but is equivalent to other versions available in the kernels.
'''

def word_match_share(x, stops=None):
    '''
    The much-loved word_match_share feature.

    Args:
        x: source data with question1/2
        stops: set of stop words to remove; default is None for nltk stops
        
    Returns:
        the ever-popular word_match_share feature as a pandas Series
    '''
    if stops is None:
        stops = set(stopwords.words('english'))
    q1 = x.question1.fillna(' ').str.lower().str.split()
    q2 = x.question2.fillna(' ').str.lower().str.split()
    q1 = q1.map(lambda l : set(l) - stops)
    q2 = q2.map(lambda l : set(l) - stops)
    q = pd.DataFrame({'q1':q1, 'q2':q2})
    q['len_inter'] = q.apply(lambda row : len(row['q1'] & row['q2']), axis=1)
    q['len_tot'] = q.q1.map(len) + q.q2.map(len)
    return (2 * q.len_inter / q.len_tot).fillna(0)


'''
To make our model, all we do is group on rounded values of word_match_share, 
and then count the number of positives and total values. We return these as dicts 
from rounded word_match_share values to the counts.    
'''

def get_stats(tr, c=10):
    '''
    Compute word_match_share on <tr>, bin the values and average the targets.
    Returns a dict from rounded word_match_share statistics to probabilities.

    Args:
        tr: pandas DataFrame with question1/2 in it
        c: word shares are rounded to whole numbers after multiplying by c.
            The default is 10.

    Returns:
        2 dicts from rounded word_match_share values to counts and positives
    '''
    x = tr[['is_duplicate']].copy()
    wms = word_match_share(tr)
    x['round_wms'] = (c * wms).round()
    gp = x.groupby('round_wms')
    pos = gp.is_duplicate.sum().to_dict()
    cts = x.round_wms.value_counts().to_dict()
    return pos, cts

'''
Let's make the model... We'll use 101 bins (0...100) instead of 11; it scores a little better.
'''    

probs, cts = get_stats(tr, c=100)

'''
Predicting involves getting rounded word_match_share values for the test set and mapping the 
counts from the step above to those values. We also use a bit of Laplace smoothing to get rid 
of 0 or 1 probabilities and fill missing values.
'''

def apply_stats(te, pos, cts, v_pos, vss, c=10):
    '''
    Applies the stats from get_stats to the test frame.

    Args:
        te: test data frame
        pos: dict of { hash(question) : positive count } for train
        cts: dict of { hash(question) : occurance counts } for train
        v_pos: number of virtual positives for smoothing (can be non-integer)
        vss: virtual sample size for smoothing (can be non-integer)
        c: word shares are rounded to whole numbers after multiplying by c.
            The default is 10.

    Returns:
        pandas Series of probabilities.
    '''
    wms = word_match_share(te)
    round_wms = (c * wms).round()
    te_pos = round_wms.map(lambda x : pos.get(x, 0))
    te_cts = round_wms.map(lambda x : cts.get(x, 0))
    prob = (te_pos + v_pos) / (te_cts + vss)
    return prob


raw_pred = apply_stats(te, probs, cts, 1, 3, c=100)    


'''
There is one more thing we need to do to get a decent score. 
The data that backs the public LB has a different mean that the training data. 
I discussed that in another kernel: How many 1's are in the public LB? 
(https://www.kaggle.com/davidthaler/quora-question-pairs/how-many-1-s-are-in-the-public-lb). 
We have to account for that somehow. I do it here by shifting the predictions on the log-odds scale.
'''

def mean_label_adjust(sub, target, current):
    '''
    Uses case-control type log-odds additive shift to correct label bias.

    Args:
        sub: the submission frame
        target: the target value for the mean label
        current: the current OOB or CV mean label, 
                or mean label of training data

    Returns:
        copy of sub with mean prediction shifted.
    '''
    target_log_odds = np.log(target / (1 - target))
    current_log_odds = np.log(current / (1 - current))
    out = sub.copy()
    out['log_odds'] = np.log(out.is_duplicate/ (1 - out.is_duplicate))
    out['adj_log_odds'] = out.log_odds - current_log_odds + target_log_odds
    out.is_duplicate = 1 / (1 + np.exp(-out.adj_log_odds))
    return out[['test_id', 'is_duplicate']]

# To use this, we need a submission frame. I'm shifting to 0.175 from 0.37, which seems to be a little better.

sub = te[['test_id']].copy()
sub['is_duplicate'] = raw_pred
sub = mean_label_adjust(sub, 0.175, 0.37)
sub.to_csv('no_ml_model.csv', index=False, float_format='%.6f')