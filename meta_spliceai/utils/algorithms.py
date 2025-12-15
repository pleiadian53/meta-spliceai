import collections
import re
import sys
import time
# import utils_sys as utils
import heapq
from operator import itemgetter


class Graph(object):
    def __init__(adjlist=None, vertices=None, V=None): 
        self.adj = {}
        self.V = V
        self.path = {}  # keep track of all reachable vertics starting from a given vertex v

        if adjlist is not None: 
            self.V = len(adjlist)
            for h, vx in adjlist.items(): 
                self.adj[h] = vx

        elif vertices is not None: 
            assert hasattr(vertices, '__iter__')
            self.V = len(vertices)
            for v in vertices: 
                self.adj[v] = []
        else: 
            assert isinstance(V, int)
            self.V = V
            for i in range(V): 
                self.adj[i] = []
    def DFS(x): 
        pass 
    def DFStrace(): 
        pass

# class Tracker(object): 
#     def __init__(self, parameters: tuple[str], settings={}, to_str=True, sep='_'): 
#         self.db = dict.fromkeys(sep.join(parameters) if to_str else tuple(parameters))

# def algorithm_tracker(parameters, settings, db={}): 
#     """
#     Takes in a list (or tuple) of parameters and convert them to keys 
#     """
#     pass

def least_common(array, to_find=None):
    # import heapq 
    # from operator import itemgetter
    counter = collections.Counter(array)
    if to_find is None:
        return sorted(counter.items(), key=itemgetter(1), reverse=False)
    return heapq.nsmallest(to_find, counter.items(), key=itemgetter(1))

def tokenize(string):
    """Convert string to lowercase and split into words (ignoring
    punctuation), returning list of words.
    """
    # '\w+' does not work well for codes with special chars such as '.' as part of the 'word'
    return re.findall(r'([-0-9a-zA-Z_:.]+)', string.lower())  


def find_ngrams(input_list, n=3):
    """
    Example
    -------
    input_list = ['all', 'this', 'happened', 'more', 'or', 'less']

    """
    return zip(*[input_list[i:] for i in range(n)])

def count_given_ngrams(seqx, ngrams, partial_order=True):
    """
    Count numbers of occurrences of ngrams in input sequence (seqx, a list of a list of ngrams)

    Related
    -------
    count_given_ngrams2()

    Output
    ------
    A dictionary: n-gram -> count 
    """    

    # usu. the input ngrams have the same length 
    ngram_tb = {1: [], }
    for ngram in ngrams: # ngram is in tuple form 
        if isinstance(ngram, tuple): 
            length = len(ngram)
            if not ngram_tb.has_key(length): ngram_tb[length] = []
            ngram_tb[length].append(ngram)
        else: # assume to be unigrams 
            ngram_tb[1].append(ngram)
            
    ng_min, ng_max = min(ngram_tb.keys()), max(ngram_tb.keys())
    if partial_order:

        # evaluate all possible n-grams 
        counts = count_ngrams2(seqx, min_length=ng_min, max_length=ng_max, partial_order=True)

        counts_prime = {}
        for n, ngx in ngram_tb.items(): # n in n-gram
            if counts.has_key(n): 
                for ngram in ngx: # query each desired ngram 
                    # if n == 1: print '> unigram: %s' % ngram
                    # sorted('x') == sorted(('x', )) == ['x'] =>  f ngram is a unigram, can do ('e', ) or 'e'
                    counts_prime[ngram] = counts[n][tuple(sorted(ngram))] 
            else: 
                for ngram in ngx: 
                    counts_prime[ngram] = 0 
    else: 
        counts = count_ngrams2(seqx, min_length=ng_min, max_length=ng_max, partial_order=False)
        
        counts_prime = {}
        for n, ngx in ngram_tb.items(): # n in n-gram
            if counts.has_key(n): 
                for ngram in ngx: # query each desired ngram 
                    counts_prime[ngram] = counts[n][tuple(ngram)]
            else: 
                for ngram in ngx: 
                    counts_prime[ngram] = 0 

    return counts_prime  # n-gram -> count

def count_given_ngrams2(seqx, ngrams, partial_order=True):
    """
    Count numbers of occurrences of ngrams in input sequence (seqx, a list of a list of ngrams)

    Output
    ------
    A dictionary: n (as in ngram) -> ngram -> count 
    """
    # the input ngrams may or may not have the same length 
    ngram_tb = {1: [], }
    for ngram in ngrams: # ngram is in tuple form 
        if isinstance(ngram, tuple): 
            length = len(ngram)
            if not ngram_tb.has_key(length): ngram_tb[length] = []
            ngram_tb[length].append(ngram)
        else: # assume to be unigrams 
            assert isinstance(ngram, str)
            ngram_tb[1].append(ngram)
            
    # print('verify> ngram_tb:\n%s\n' % ngram_tb) # utils.sample_hashtable(ngram_tb, n_sample=10))

    ng_min, ng_max = min(ngram_tb.keys()), max(ngram_tb.keys())
    if partial_order:

        # evaluate all possible n-grams 
        counts = count_ngrams2(seqx, min_length=ng_min, max_length=ng_max, partial_order=True)

        counts_prime = {}
        for n, ngx in ngram_tb.items(): # n in n-gram
            if not counts_prime.has_key(n): counts_prime[n] = {} 
            if counts.has_key(n):
                for ngram in ngx: # query each desired ngram 
                    # if n == 1: print '> unigram: %s' % ngram
                    # sorted('x') == sorted(('x', )) == ['x'] =>  f ngram is a unigram, can do ('e', ) or 'e'
                    counts_prime[n][ngram] = counts[n][tuple(sorted(ngram))] 
            else: 
                for ngram in ngx: 
                    counts_prime[n][ngram] = 0 
    else: 
        counts = count_ngrams2(seqx, min_length=ng_min, max_length=ng_max, partial_order=False)
        
        counts_prime = {}
        for n, ngx in ngram_tb.items(): # n in n-gram
            if not counts_prime.has_key(n): counts_prime[n] = {} 
            if counts.has_key(n): 
                for ngram in ngx: # query each desired ngram 
                    # assert isinstance(ngram, tuple), "Ngram is not a tuple: %s" % str(ngram)
                    counts_prime[n][ngram] = counts[n][tuple(ngram)]
            else: 
                for ngram in ngx: 
                    counts_prime[n][ngram] = 0 

    return counts_prime  # n (as n-gram) -> counts (ngram -> count)

def count_ngrams2(lines, min_length=2, max_length=4, **kargs): 
    def eval_sequence_dtype(): 
        if not lines: 
            return False # no-op
        if isinstance(lines[0], str): # ['a b c d', 'e f', ]
            return False
        elif hasattr(lines[0], '__iter__'): # [['a', 'b'], ['c', 'd', 'e'], ]
            return True
        return False

    is_partial_order = kargs.get('partial_order', True)
    lengths = range(min_length, max_length + 1)    
    
    # is_tokenized = eval_sequence_dtype()
    seqx = []
    for line in lines: 
        if isinstance(line, str): # not tokenized  
            seqx.append([word for word in tokenize(line)])
        else: 
            seqx.append(line)
    
    # print('count_ngrams2> debug | seqx: %s' % seqx[:5]) # list of (list of codes)
    if not is_partial_order:  # i.e. total order 
        # ordering is important

        # this includes ngrams that CROSS line boundaries 
        # return count_ngrams(seqx, min_length=min_length, max_length=max_length) # n -> counter (of n-grams)

        # this counts ngrams in each line independently 
        counts = count_ngrams_per_seq(seqx, min_length=min_length, max_length=max_length) # n -> counter (of n-grams)
        return {length: counts[length] for length in lengths}

    # print('> seqx:\n%s\n' % seqx)
    # print('status> ordering NOT important ...')
    
    counts = {}
    for length in lengths: 
        counts[length] = collections.Counter()
        # ngrams = find_ngrams(seqx, n=length)  # list of n-grams in tuples
        if length == 1: 
            for seq in seqx: 
                counts[length].update([(ugram, ) for ugram in seq])
        else: 
            for seq in seqx:  # use sorted n-gram to standardize its entry since ordering is not important here
                counts[length].update( tuple(sorted(ngram)) for ngram in find_ngrams(seq, n=length) ) 

    return counts

def count_ngrams_per_line(**kargs):
    return count_ngrams_per_seq(**kargs)
def count_ngrams_per_seq(lines, min_length=1, max_length=4): # non boundary crossing  
    def update(ngrams):
        # print('> line = %s' % single_doc)
        for n, counts in ngrams.items(): 
            # print('  ++ ngrams_total: %s' % ngrams_total)
            # print('      +++ ngrams new: %s' % counts)
            ngrams_total[n].update(counts)
            # print('      +++ ngrams_total new: %s' % ngrams_total)

    lengths = range(min_length, max_length + 1)
    ngrams_total = {length: collections.Counter() for length in lengths}

    doc_boundary_crossing = False
    if not doc_boundary_crossing: # don't count n-grams that straddles two documents
        for line in lines: 
            nT = len(line)
            # print(' + line=%s, nT=%d' % (line, nT))
            single_doc = [line]

            # if the line length, nT, is smaller than max_length, will miscount
            ngrams = count_ngrams(single_doc, min_length=1, max_length=min(max_length, nT))
            update(ngrams) # update total counts
    else: 
        raise NotImplementedError

    return ngrams_total

def count_ngrams(lines, min_length=1, max_length=4): 
    """
    Iterate through given lines iterator (file object or list of
    lines) and return n-gram frequencies. The return value is a dict
    mapping the length of the n-gram to a collections.Counter
    object of n-gram tuple and number of times that n-gram occurred.
    Returned dict includes n-grams of length min_length to max_length.

    Use this only when (strict) ordering is important; otherwise, use count_ngrams2()

    Input
    -----
    lines: [['x', 'y', 'z'], ['y', 'x', 'z', 'u'], ... ]
    """
    def add_queue():
        # Helper function to add n-grams at start of current queue to dict
        current = tuple(queue)
        for length in lengths:
            if len(current) >= length:  # count n-grams up to length those in queue
                ngrams[length][current[:length]] += 1  # ngrams[length] => counter
    def eval_sequence_dtype(): 
        if not lines: 
            return False # no-op
        if isinstance(lines[0], str): 
            return False
        elif hasattr(lines[0], '__iter__'): 
            return True
        return False

    lengths = range(min_length, max_length + 1)
    ngrams = {length: collections.Counter() for length in lengths}
    queue = collections.deque(maxlen=max_length)

    # tokenized or not? 
    is_tokenized = eval_sequence_dtype()
    # print('> tokenized? %s' % is_tokenized)

    # Loop through all lines and words and add n-grams to dict
    if is_tokenized: 
        # print('input> lines: %s' % lines)
        for line in lines:
            for word in line:
                queue.append(word)
                if len(queue) >= max_length:
                    add_queue()  # this does the counting
            # print('+ line: %s\n+ngrams: %s' % (line, ngrams))
    else: 
        for line in lines:
            for word in tokenize(line):
                queue.append(word)
                if len(queue) >= max_length:
                    add_queue()

    # Make sure we get the n-grams at the tail end of the queue
    while len(queue) > min_length:
        queue.popleft()
        add_queue()
        # print('+ line: %s\n+ngrams: %s' % (line, ngrams))

    return ngrams

def check_boundary(lines, ngram_counts):
    # def isInDoc(ngstr): 
    #     for line in lines: 
    #         linestr = sep.join(str(e) for e in line)
    #         if linestr.find(ngstr) >= 0: 
    #             return True 
    #     return False

    # sep = ' ' 
    # for n, counts in ngram_counts: 
    #     counts_prime = []  # only keep those that do not cross line boundaries
    #     crossed = set()
    #     for ngr, cnt in counts: 

    #         # convert to string 
    #         ngstr = sep.join([str(e) for e in ngr])
    #         if isInDoc(ngstr): 
    #             counts_prime[]
    raise NotImplementedError
    # return ngram_counts  # new ngram counts


# [algorithm]
def lcs(S,T):
    """
    Find longest common substring
    """
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])

    return lcs_set

def size_hashtable(adict): 
    return sum(len(v) for k, v in adict.items())

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

def pair_to_hashtable(zlist, key=1):
    vid = 1-key 
    adict = {}
    for e in zlist:
        if not e[key] in adict:   
            adict[e[key]] = [] 
        adict[e[key]].append(e[vid])    
    return adict

def sample_hashtable(hashtable, n_sample=10):
    import random, gc, copy
    from itertools import cycle

    n_sampled = 0
    tb = copy.deepcopy(hashtable)
    R = tb.keys(); random.shuffle(R)
    nT = sum([len(v) for v in tb.values()])
    print('sample_hashtable> Total keys: %d, members: %d' % (len(R), nT))
    
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

def dictToList(adict):
    lists = []
    for k, v in nested_dict_iter(adict): 
        alist = []
        if not hasattr(k, '__iter__'): k = [k, ]
        if not hasattr(v, '__iter__'): v = [v, ]
        alist.extend(k)
        alist.extend(v)
        lists.append(alist)
    return lists

def nested_dict_iter(nested):
    import collections

    for key, value in nested.items():  # nested.iteritems() => in Python 3, use items()
        if isinstance(value, collections.Mapping):
            for inner_key, inner_value in nested_dict_iter(value):
                yield inner_key, inner_value
        else:
            yield key, value

def dictSize(adict): # yeah, size matters  
    return len(list(nested_dict_iter(adict)))
def size_dict(adict): 
    """

    Note
    ----
    1. size_hashtable()
    """
    return len(list(nested_dict_iter(adict)))


def partition(lst, n):
    """
    Partition a list into almost equal intervals as much as possible. 
    """
    q, r = divmod(len(lst), n)
    indices = [q*i + min(i, r) for i in xrange(n+1)]
    return [lst[indices[i]:indices[i+1]] for i in xrange(n)]

def divide_interval(total, n_parts):
    pl = [0] * n_parts
    for i in range(n_parts): 
        pl[i] = total // n_parts    # integer division

    # divide up the remainder
    r = total % n_parts
    for j in range(r): 
        pl[j] += 1

    return pl 

def combinations(iterable, r):
    """

    Memo
    ----
    itertools.combinations

    Reference
    ---------
    https://docs.python.org/2/library/itertools.html#itertools.combinations
    """
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)


def print_most_frequent(ngrams, num=10):
    """Print num most common n-grams of each length in n-grams dict."""
    for n in sorted(ngrams):
        print('----- {} most common {}-grams -----'.format(num, n))
        for gram, count in ngrams[n].most_common(num):
            print('{0}: {1}'.format(' '.join(gram), count))
        print('')

def t_priority_queue(): 
    import platform
    import heapq

    try:
        import Queue as Q  # ver. < 3.0
    except ImportError:
        print("> import queue | python version %d" % platform.python_version())
        import queue as Q

    q = Q.PriorityQueue()
    q.put((10,'ten'))
    q.put((1,'one'))
    q.put((5,'five'))
    while not q.empty():
        print ('%s, ' % q.get())

    print('info> try heapq module ...')
    
    heap = []
    heapq.heappush(heap, (-1.5, 'negative one'))
    heapq.heappush(heap, (1, 'one'))
    heapq.heappush(heap, (10, 'ten'))
    heapq.heappush(heap, (5.7,'five'))
    heapq.heappush(heap, (100.6, 'hundred'))

    for x in heap:
        print ('%s, ' % x)
    print()

    heapq.heappop(heap)

    for x in heap:
        print('{0},'.format(x)) # print x,   
    print()

    # the smallest
    print('info> smallest: %s' % str(heap[0]))

    smallestx = heapq.nsmallest(2, heap)  # a list
    print('info> n smallest: %s, type: %s' % (str(smallestx), type(smallestx)))

    return

def split(alist, n):
    n = min(n, len(alist)) # don't create empty buckets in scenarios like list(split(range(X, Y))) where X < Y
    k, m = divmod(len(alist), n)
    return (alist[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def multireplace(string, replacements, ignore_case=False):
    """
    Given a string and a replacement map, it returns the replaced string.

    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :param bool ignore_case: whether the match should be case insensitive
    :rtype: str

    """
    if not replacements:
        # Edge case that'd produce a funny regex and cause a KeyError
        return string
    
    # If case insensitive, we need to normalize the old string so that later a replacement
    # can be found. For instance with {"HEY": "lol"} we should match and find a replacement for "hey",
    # "HEY", "hEy", etc.
    if ignore_case:
        def normalize_old(s):
            return s.lower()

        re_mode = re.IGNORECASE

    else:
        def normalize_old(s):
            return s

        re_mode = 0

    replacements = {normalize_old(key): val for key, val in replacements.items()}
    
    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
    rep_sorted = sorted(replacements, key=len, reverse=True)
    rep_escaped = map(re.escape, rep_sorted)
    
    # Create a big OR regex that matches any of the substrings to replace
    pattern = re.compile("|".join(rep_escaped), re_mode)
    
    # For each match, look up the new string in the replacements, being the key the normalized old string
    return pattern.sub(lambda match: replacements[normalize_old(match.group(0))], string)

def drop_segment(input_str, start, end):
    
    
    # ... group: c-----------c
    # ... span: (41, 54)
    # ...... 41: c, -
    # ...... 54: c, c
  
    
    # print(result)

    return 

def drop_introns_customized(seq='', marker_seq='', n_retained_introns=100, **kargs): 
    return retain_partial_introns(seq=seq, marker_seq=marker_seq, n_retained_introns=n_retained_introns, **kargs)
def retain_partial_introns_prototype(seq='', marker_seq='', intron='-', n_retained_introns=100, **kargs): 
    """
    
    Memo
    ----
    1. Regex 
        https://www.regular-expressions.info/lookaround.html
    """
    import numpy as np
    if not marker_seq or not seq: 
        # Use a test case
        print("> Missing inputs, using a test case ...") 
        # -----------
        marker_seq = "5555555555555555555555BBBccccccccccccccccc-----------cccccccccccccccccccccccccc----cccccccccccccc-------------------ccccccccccc-----------------------------EEEcccccccccccccccxxx33333333333333333333333333"
        seq = ''.join(np.random.choice(['A', 'T', 'G', 'C'], len(marker_seq)))
    assert len(seq) == len(marker_seq)
    # coo_drop_list = []
    # pattern = '[B]{3}' # stop codon 
    
    intron = kargs.get('intron', '-') 
    # pattern = f'[^{intron}][{intron}]+[^{intron}]' # intron boundary, assuming an intron is repr by '-'
    pattern = f'(?<=[^{intron}])[{intron}]+(?=[^{intron}])' # using positive lookbehind and positive lookahead 

    # A. Find all occurrences of a pattern, and then try to replace them all at the same time (tricky)
    replace_candidates = []
    intron_lengths = []
    intron_coos = []
    new_seq = new_marker_seq = ''
    new_segments = []
    # n_retained_introns = 100
    n_segments = 0
    last_three_prime_segment = ''
    for occurrence in re.finditer(r'{}'.format(pattern), marker_seq): 
        print(occurrence.group()) 
        print(occurrence.span())

        start, end = occurrence.span()
        print(f"> *[start-1]: {marker_seq[start-1]}, *[start]: {marker_seq[start]}, *[end-1]:{marker_seq[end-1]}, *[end]: {marker_seq[end]}")
        print(marker_seq[start: end]) 
        assert marker_seq[start: end] == occurrence.group() # ... ok
        
        intron_len = end - start 
        intron_lengths.append(intron_len)

        new_seq = marker_seq[:start] + marker_seq[end:]  # BBB removed 
        assert len(marker_seq) - len(new_seq) == intron_len
        # print(new_seq[:100])
        
        five_prime_segment = marker_seq[:start]
        intron_segment = marker_seq[start: end] # e.g. introns
        three_prime_segment = marker_seq[end:]
        assert len(five_prime_segment) + len(intron_segment) + len(three_prime_segment) == len(marker_seq)
        
        intron_start, intron_end = start, end # current intron start and end
        # Adjusted 5' segment 
        if len(intron_coos) > 0: 
            last_intron_start, last_intron_end = intron_coos[-1]
            five_prime_segment = marker_seq[last_intron_end:intron_start]
        last_three_prime_segment = three_prime_segment # only meant to capture the last segment; the earlier segments would contain introns

        new_segments.append(five_prime_segment)
        new_segments.append(intron_segment) # or new intron_segment
        
        intron_coos.append((start, end)) # keep track of the intron segment positions up to the current intron segment

        # marker_seq[start:start+n_retained_introns]
        # coo_drop_list.append((start, end)) # substrings between start and end should be removed
        replace_candidates.append(occurrence.group())

        n_segments += 1
    
    if len(intron_lengths) > 0:
        new_segments.append(last_three_prime_segment) # append the last 3' end segment

        print(f"> new sequence segments:\n{new_segments}\n")
        total_len = 0
        for segment in new_segments: 
            total_len += len(segment)
        assert total_len == len(marker_seq)
        
        new_seq = ''.join(new_segments)
        # drop_segment(marker_seq)
        # print(coo_drop_list)
        # for coo in coo_drop_list: 
        #     pass
        print(f"> len of joined new seq: {len(new_seq)} =?= {len(marker_seq)}")

        # Apply multiple replacements simultaneously? 
        rdict = {candidate: "" for candidate in replace_candidates}
        new_seq_intron_dropped = multireplace(marker_seq, replacements=rdict, ignore_case=False)
        print(new_seq_intron_dropped)
        print(f"> length: original={len(marker_seq)}, new={len(new_seq_intron_dropped)}, delta={len(marker_seq)-len(new_seq_intron_dropped)}")
        print(f"... sum(dropped segments): {np.sum(intron_lengths)}")
    else: 
        new_seq = marker_seq

    print(f"> total length: {total_len} =?= {len(marker_seq)}")

    # B. Search the pattern one at a time, followed by substring replacement and then do the same with the next search
    # new_seq = ''
    # occurrence = re.search(r'{}'.format(pattern), marker_seq)
    # print(occurrence.group()) # c------c
    # print(occurrence.span())  # ^
    # start, end = occurrence.span()  # start: 
    # print(f"> end points: marker_seq[start]: {marker_seq[start]}, *[end]: {marker_seq[end-1]}") # c, c 
    # intron_segment = marker_seq[start+1: end-1]
    # print(f"... intron seg (len={len(intron_segment)}): {intron_segment}")
    # new_segment = '---'
    # print(marker_seq[:start])
    # new_seq = marker_seq[:start] + new_segment + marker_seq[end-1:]
    # print(new_seq)

    return

def demo_find_substr_by_regex(): 
    """
    
    Memo
    ----
    1. Find occurrences of substrings 
       - https://pynative.com/python-regex-findall-finditer/
    """
    import re

    target_string = "Emma is a basketball player who was born on June 17, 1993. She played 112 matches with scoring average 26.12 points per game. Her weight is 51 kg."
    result = re.findall(r"\d+", target_string)

    # print all matches
    print("Found following matches")
    print(result)

    marker_seq = "5555555555555555555555BBBccccccccccccccccc-----------cccccccccccccccccccccccccccccccccccccccc-------------------ccccccccccc-----------------------------EEEcccccccccccccccxxx33333333333333333333333333"
    result = re.findall(r'\w[-]+\w', marker_seq)
    print(result)

    # In some scenarios, the number of matches is high, and you could risk filling up your memory ...
    occurrences = re.finditer(r'\w[-]+\w', marker_seq)

    # print all match object
    intron_partial_length = 100
    for i, occurrence in enumerate(occurrences):
        # print each re.Match object
        print("> [{}]: {}".format(i, occurrence))
        
        # extract each matching number
        print("... group: {}".format(occurrence.group()))
        print('... span: {}'.format(occurrence.span()))

        start, end = occurrence.span()
        print(f"...... {start}: {marker_seq[start]}, {marker_seq[start+1]}")
        print(f"...... {end}: {marker_seq[end]}, {marker_seq[end]}")

        intron_segment = marker_seq[start+1:end]
        intron_start = start+1 # inclusive
        intron_end = end  # exclusive 

        intron_len = end - start - 1  # ends exclusive
        print(f"...... len(intron): {intron_len}")
        assert len(intron_segment) == intron_len

        # include partial introns 

    print(); print('-' * 90)
    print("> Search pattern in marker sequence ...")
    res = re.search(r'\w[-]+\w', marker_seq)
    print(res.span()) # but this only gives the first occurrence

    print(); print('-' * 90)
    # Find words with repeated characters 
    target_string = "Jessa Erriika"
    # This '\w' matches any single character
    # and then its repetitions (\1*) if any.
    matcher = re.compile(r"(\w)\1+")
    for match in matcher.finditer(target_string):
        print(match.group(), end=", ")
    print()


    return


def test(): 
    
    # split a list of elements into N (approx) equal parts 
    # parts = list(split(range(11), 3))
    # print(parts)

    # Substring matching 
    # demo_find_substr_by_regex()
    retain_partial_introns_prototype()

    return

if __name__ == "__main__": 
    test()