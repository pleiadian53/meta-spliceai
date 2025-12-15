# encoding: utf-8
import statistics 
import os, sys, re
import random
import collections
import numpy as np 

# import matplotlib
# matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
# from matplotlib import pyplot as plt

from pandas import DataFrame, Series
import pandas as pd 
from bisect import bisect

###############################################################################################################
#
# 
#
###############################################################################################################
"""

Reference
---------
1. sampling methods in python
    https://people.duke.edu/~ccc14/sta-663/ResamplingAndMonteCarloSimulations.html

"""

def sample_by_category(df, cat_col, *, n=None, replace=False, verbose=1):  
    uniq_values = df[cat_col].unique()
    n_uvals = len(uniq_values)

    if n is None: n = n_uvals
    sampled_values = np.random.choice(uniq_values, n, replace=replace)
    return df.loc[df[cat_col].isin(sampled_values)]

def sample_by_proportion(df, target_col, n=None, replace=False, verbose=1): 
    """

    Memo
    ----
        - find class weights proportional to their sample sizes
            
           df.value_counts(class_col, normalize=True)

    Todo 
    ----
        - Compensate for indivisible sample sizes 
        
    """
    if n is None: n = df.shape[0]
    N0 = n
    # if n > df.shape[0]: 
    #     replace=True

    weights = df[target_col].value_counts(normalize=True)
    sizes = (weights * n).astype(int)
    # print(f"> sizes: {sizes}")

    dfx = []
    for v, dfg in df.groupby(target_col): 
        # print(f"> sampling dfg group={v} with sample size: {sizes[v]} ...")
        dfx.append(dfg.sample(n=sizes[v], replace=replace)) 
    df_subset = pd.concat(dfx, ignore_index=True)

    # Alternatively 
    # df_subset = df.groupby(target_col).apply(lambda s: s.sample(int(n * weights[s.name]))).droplevel(target_col)
    if verbose: 
        print(f"> Sampling by proportion: N0={N0} => {df_subset.shape[0]}")

    return df_subset

def weighted_choice(choices):
    """
    choices: 
       [("WHITE",90), ("RED",8), ("GREEN",2)]
    """
    values, weights = zip(*choices)
    total = 0
    cum_weights = []
    for w in weights:
        total += w
        cum_weights.append(total)
    x = random.random() * total
    i = bisect(cum_weights, x)
    return values[i]

def estimate_confidence_interval0(x, n_rep=1000, lower=None, upper=97.5): 
    n = len(x)  # dim(x)
    xb = np.random.choice(x, (n, n_rep), replace=True)  # sampling with replacement n times
    mb = xb.mean(axis=0)

    # estimate confidence interval
    mb.sort()
    if lower is None: lower = 100.0-upper
    return np.percentile(mb, [lower, upper])

def bootstrap_resample(x, n=None, random_state=1, verbose=False): 
    """

    Reference
    ---------
    1. https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/
    """
    from sklearn.utils import resample   # scikit-learn bootstrap
    # data sample
    data = x
    # prepare bootstrap sample
    boot = resample(data, replace=True, n_samples=n, random_state=random_state)
    if verbose: print('> Bootstrap Sample: %s' % boot)
    
    # out of bag observations
    oob = [x for x in data if x not in boot]
    if verbose: print('> OOB Sample: %s' % oob)

    return (boot, oob)

def bootstrap_resample2(X, n=None, y=None, all_labels_present=True, n_cycles=20):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    import collections
    if isinstance(X, pd.Series):
        X = X.copy()
        X.index = range(len(X.index))
    else: 
         X = np.array(X) # need to use array/list to index elements
    if n is None: n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int) # e.g. array([ 8410, 11437, 87128, ..., 75103,  5866, 44852])
    X_resample = np.array(X[resample_i])  # TODO: write a test demonstrating why array() is important

    if y is not None:
        labels = np.unique(y)
        n_labels = len(labels)
        assert len(y) == len(X)

        y_resample = np.array(y[resample_i]) 
        if all_labels_present: 
            n_labels_resample = len(np.unique(y_resample))
            while True:
                if n_labels_resample == n_labels: break 
                # need to resample again 
                if j > n_cycles: 
                    print('bootstrap> after %d cycles of resampling, still could not have all labels present.')
                    ac = collections.Counter(y)
                    print('info> class label counts:\n%s\n' % ac) 
                    break

                resample_i = np.floor(np.random.rand(n)*len(X)).astype(int) # e.g. array([ 8410, 11437, 87128, ..., 75103,  5866, 44852])
                X_resample = np.array(X[resample_i])  # TODO: write a test demonstrating why array() is important
                y_resample = np.array(y[resample_i]) 

                j += 1 
            # assert np.unique(y_resample) == n_labels

        return (X_resample, y_resample)

    return X_resample

def ci(scores, low=0.05, high=0.95):
    sorted_scores = np.array(scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(low * len(sorted_scores))]
    confidence_upper = sorted_scores[int(high * len(sorted_scores))]
    # print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
    #     confidence_lower, confidence_upper)) 
    return (confidence_lower, confidence_upper)

def ci2(scores, low=0.05, high=0.95, mean=None):
    std = statistics.stdev(scores) 
    mean_score = np.mean(scores)  # bootstrap sample mean
    if mean is None: mean = mean_score

    sorted_scores = np.array(scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(low * len(sorted_scores))]
    confidence_upper = sorted_scores[int(high * len(sorted_scores))]
    middle = (confidence_upper+confidence_lower)/2.0  # assume symmetric

    print('ci2> mean score: %f, middle: %f' % (mean_score, middle))
    # mean = sorted_scores[int(0.5 * len(sorted_scores))]
    # print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
    #     confidence_lower, confidence_upper)) 

    if confidence_upper > 1.0: 
        print('ci2> Warning: upper bound larger than 1.0! %f' % confidence_upper)
        confidence_upper = 1.0

    # this estimate may exceeds 1 
    delminus, delplus = (mean-confidence_lower, confidence_upper-mean)

    return (confidence_lower, confidence_upper, delminus, delplus, std)

def ci3(scores, low=0.05, high=0.95):
    if isinstance(scores[0], int): 
        scores = [float(e) for e in scores]
    sorted_scores = np.array(scores)
    sorted_scores.sort()
    mean_score = np.mean(scores)  # bootstrap sample mean
    se = statistics.stdev(scores) # square root of sample variance, standard error

    confidence_lower = sorted_scores[int(low * len(sorted_scores))]
    confidence_upper = sorted_scores[int(high * len(sorted_scores))]
    # print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
    #     confidence_lower, confidence_upper)) 
    return (mean_score, se, confidence_lower, confidence_upper)

def estimate_confidence_interval(scores, low=0.05, high=0.95):
    if isinstance(scores[0], int): 
        scores = [float(e) for e in scores]

    ret = {}
    sorted_scores = np.array(scores)
    sorted_scores.sort()
    ret['mean'] = np.mean(scores)  # bootstrap sample mean
    ret['median'] = np.median(scores)
    ret['se'] = ret['error'] = se = statistics.stdev(scores) # square root of sample variance, standard error

    ret['ci_low'] = ret['confidence_lower'] = confidence_lower = sorted_scores[int(low * len(sorted_scores))]
    ret['ci_high'] = ret['confidence_upper'] = confidence_upper = sorted_scores[int(high * len(sorted_scores))]
    # print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
    #     confidence_lower, confidence_upper)) 

    return ret

def sorted_interval_sampling(l, npar, reverse=False):
    """
    Arguments
    ---------
    npar: n partitions 
    """ 
    l.sort(reverse=reverse)
    avg = len(l)/float(npar)
    slist, partitions = [], []
    last = 0.0

    while last < len(l):
        partitions.append(l[int(last):int(last + avg)])
        last += avg    
    
    npar_eff = len(partitions) # sometimes 1 extra 
    # print('info> n_par: %d' % len(partitions))
    # print('\n%s\n' % partitions)
    for par in partitions:
        slist.append(random.sample(par, 1)[0])
        
    # 0, 1, 2, 3, 4, 5 => n=6, 6/2=3, 6/2-1=2 
    # 0, 1, 2, 3, 4    => n=5, 5/2=2 
    if npar_eff > npar: 
        assert npar_eff - npar == 1
        del slist[npar_eff/2]

    assert len(slist) == npar
    # for par in [l[i:i+n] for i in xrange(0, len(l), n)]: 
    #     alist.append(random.sample(par, 1)[0])
    return slist

# sampling with datastruct 
def sample_dict(adict, n_sample=10): 
    """
    Get a sampled subset of the dictionary. 
    """
    import random 
    keys = adict.keys() 
    n = len(keys)
    keys = random.sample(keys, min(n_sample, n))
    return {k: adict[k] for k in keys} 

def sample_subset(x, n_sample=10):
    if len(x) == 0: return x
    if isinstance(x, dict): return sample_dict(x, n_sample=n_sample)
    
    # assume [(), (), ] 
    return random.sample(x, n_sample)

def sample_cluster(cluster, n_sample=10): 
    """
    Input
    -----
    cluster: a list of cluster indices 
             e.g. 3 clusters 7 data points [0, 1, 1, 2, 2, 0, 0] 

    """
    n_clusters = len(set(cluster))
    hashtb = {cid: [] for cid in cluster}

    for i, cid in enumerate(cluster): 
        hashtb[cid].append(i)      # cid to positions        
 
    return sample_hashtable(hashtb, n_sample=n_sample)

def sample_hashtable(hashtable, n_sample=10):
    import random, gc, copy
    from itertools import cycle

    n_sampled = 0
    tb = copy.deepcopy(hashtable)
    R = tb.keys(); random.shuffle(R) # shuffle elements in R inplace 
    nT = sum([len(v) for v in tb.values()])
    print('sample_hashtable> Total keys: %d, members: %d' % (len(R), nT))

    if nT < n_sample:
        print('warning> size of hashtable: %d < n_sample: %d' % (nT, n_sample))
        n_sample = nT
    
    n_cases = n_sample 
    candidates = set()

    for e in cycle(R):
        if n_sampled >= n_cases or len(candidates) >= nT: break 
        entry = tb[e]
        if entry: 
            v = random.sample(entry, 1)
            candidates.update(v)
            entry.remove(v[0])
            n_sampled += 1

    return candidates


def divide_interval(total, n_parts):
    pl = [0] * n_parts
    for i in range(n_parts): 
        pl[i] = total // n_parts    # integer division

    # divide up the remainder
    r = total % n_parts
    for j in range(r): 
        pl[j] += 1

    return pl 

def sample_class(X, y=None, n_samples=1000, replace=False, uniform=False): 
    """
    Input
    -----
    y: labels 
       use case: if labels are cluster indices (after running a clustering algorithm), then 
                 this funciton essentially performs a cluster sampling
    """
    N = len(X)
    if n_samples > N: 
        print(f"(sample_class) Warning: Requested sample size, {n_samples}, is greater than the total sample size N={N}")
    n_samples = min(N, n_samples)

    if y is None: 
       idx = np.random.choice(range(X.shape[0]), n_samples, replace=False)
       return X[idx], None

    assert X.shape[0] == len(y)
    labels = list(set(y))
    n_labels = len(labels)
    # print(f"(sample_class) labels: {labels}, y:\n{y}\n")

    if uniform: 
        n_subsets = divide_interval(n_samples, n_parts=n_labels) # 10 => [3, 3, 4]
        
        # [log] {0: 334, 1: 333, 2: 333}
        pdict = {labels[i]: n_subset for i, n_subset in enumerate(n_subsets)} # pdict: label -> subsample size
    else: 
        # sample classes proprotionally to their sizes
        pdict = collections.Counter(y)
        for label, count in collections.Counter(y).items(): 
            pdict[label] = int(n_samples * count/(N+0.0))
        Np = sum(pdict.values())
        N_delta = N - Np
        # print(f"... N_delta: {N_delta}")
        while N_delta > 0: 
            label = np.random.choice(labels, 1)[0]
            pdict[label] += 1
            N_delta -= 1
        assert sum(pdict.values()) == N, f"Class sizes summed to: {sum(pdict.values())} but N={N}"

    # print('verify> label to n_samples pdict:\n%s\n' % pdict) # ok. [log] {0: 500, 1: 500}
    tsx = []

    Xs, ys = [], []  # candidate indices
    for l, n in pdict.items(): 
        cond = (y == l)
        Xl = X[cond]
        # yl = y[cond]

        # sampling with replacement so 'n' can be larger than data size
        # idx = np.random.randint(Xl.shape[0], size=n)
        idx = np.random.choice(Xl.shape[0], size=n, replace=replace)
        # print('verify> select %d from %d instances' % (n, Xl.shape[0]))

        # print('verify> selected indices (size:%d):\n%s\n' % (len(idx), idx))
        Xs.append(Xl[idx, :])  # [note] numpy append: np.append(cidx, [4, 5])
        ys.append([l] * len(idx))
    
    assert len(Xs) == n_labels
    Xsub = np.vstack(Xs)  
    assert Xsub.shape[0] == n_samples   
    ysub = np.hstack(ys)
    assert ysub.shape[0] == n_samples

    return (Xsub, ysub)

def negative_sample(target_set, candidate_set, n=None, replace=False):
    """
    Subsample `n` elements from a candidate set (`candidate_set`) that 
    do not appear in the target set (`target_set`)
    """
    if not isinstance(target_set, (set, list, np.ndarray)): 
        target_set = set([target_set])
    if n is None: n = len(target_set)
    
    not_target_set = set(candidate_set)-set(target_set)
    N = len(not_target_set)
    if not replace: 
        assert N >= n 

    return np.random.choice(list(set(candidate_set)-set(target_set)), n, replace=replace)

def demo_class_sampling(): 
    from sklearn import datasets

    # [note] n_classes * n_clusters_per_class must be smaller or equal 2 ** n_informative
    X, y = datasets.make_classification(n_samples=3000, n_features=20,
                                    n_informative=15, n_redundant=3, n_classes=3,
                                    random_state=42)
    n_labels = len(set(y))
    print('data> dim(X): %s, y: %s > n_labels: %d' % (str(X.shape), str(y.shape), n_labels))

    Xsub, ysub = sample_class(X, y=y, n_samples=5000, replace=True)
    n_labels_sampled = len(set(ysub))
    print('sampled> dim(X): %s, y: %s > n_labels: %d' % (str(Xsub.shape), str(ysub.shape), n_labels_sampled))

    Xsub, ysub = sample_class(X, n_samples=5000, replace=True)
    n_labels_sampled = len(set(ysub))
    print('sampled> dim(X): %s, y(DUMMY): %s > n_labels: %d' % (str(Xsub.shape), str(ysub.shape), n_labels_sampled))

    return

def demo_sample_by_strata():
    
    df = pd.DataFrame([[1.1, 1.1, 1.1, 2.6, 2.5, 3.4,2.6,2.6,3.4,3.4,2.6,1.1,1.1,3.3], list('AAABBBBABCBDDD'), 
                       [1.1, 1.7, 2.5, 2.6, 3.3, 3.8,4.0,4.2,4.3,4.5,4.6,4.7,4.7,4.8], 
                       ['x/y/z','x/y','x/y/z/n','x/u','x','x/u/v','x/y/z','x','x/u/v/b','-','x/y','x/y/z','x','x/u/v/w'], 
                       ['1','3','3','2','4','2','5','3','6','3','5','1','1','1']]).T
    df.columns = ['a', 'b', 'c', 'd', 'e']
    print(f"> data shape: {df.shape}, n_uniq={len(df.b.unique())}")

    df = sample_by_category(df, cat_col='b', n=2, replace=False, verbose=True)
    print(f"> data shape: {df.shape}, n_uniq={len(df.b.unique())}")

    print(df)
    return

def demo_dataframe_sampling(): 
    from tabulate import tabulate

    alist = [0] * 20 + [1] * 80

    adict = {'a': alist, 'b': np.random.choice(range(10), len(alist))}
    df = DataFrame(adict)

    col = 'a'
    n = 50
    df_subset = sample_by_proportion(df, target_col=col, n=n)
    print(dict(df_subset.value_counts(col))); print('-' * 80)
    # print(tabulate(df_subset, headers='keys', tablefmt='psql'))

    weights = df[col].value_counts(normalize=True)
    df_subset2 = df.groupby(col).apply(lambda s: s.sample(int(n * weights[s.name]))).droplevel(col)
    print(dict(df_subset2.value_counts(col))); print('-' * 80)
    # print(tabulate(df_subset2, headers='keys', tablefmt='psql'))

    # NOTE: Below cannot guarantee the right proportion in each run, but on average, after many runs, 
    #       the sample sizes associated with `col` will be in proportional to their respective counts
    df_subset3 = df.sample(n=n)
    print(dict(df_subset3.value_counts(col)))

    return

def test(): 

    ### negative sampling 
    # subset = negative_sample([1, 3], list(range(10)))
    # print(f"[test] subset:\n{subset}\n")

    ### class sampling

    ### dataframe sampling 
    # demo_dataframe_sampling()

    demo_sample_by_strata()
 
    return

if __name__ == "__main__": 
    test()


