import os, sys, glob
import re, collections
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

"""

References 
----------

"""

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
    Check if two floating point numbers `a` and `b` are "equal" given tolerance. 

    Memo
    ----
    Python 3.5 adds the math.isclose and cmath.isclose functions
    """
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def longestCommonPrefix(s1, s2):
    i = 0
    while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
        i += 1
    return s1[:i]

def match(s1, s2):
    if not len(s1) == len(s2):
        return False
    for i in range(len(s1)):
        if not s1[i] == s2[i]:
            return False
    return True

def reverse_complement(s):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    t = ''
    for base in s:
        t = complement[base] + t
    return t

def read_genome(filename):
    genome = ''
    with open(filename, 'r') as f:
        for line in f:
            if not line[0] == '>':
                genome += line.rstrip()
    return genome
def read_fasta(filename):
    genome = ''
    with open(filename, 'r') as f:
        for line in f:
            # ignore header line with genome information
            if not line[0] == '>':
                genome += line.rstrip()
    return genome

def read_fastq(filename):
    sequences = []
    qualities = []
    with open(filename, 'r') as f:
        while True:
            f.readline()
            seq = f.readline().rstrip()
            f.readline()
            qual = f.readline().rstrip()
            if len(seq) == 0:
                break
            sequences.append(seq)
            qualities.append(qual)
    return sequences, qualities

### Sequencing Quality ### 

def QtoPhred33(Q):
    '''Turn Q into Phred+33 ASCII-­‐encoded quality'''
    return chr(Q + 33) # converts character to integer according to ASCII table

def phred33ToQ(qual):
    '''Turn Phred+33 ASCII-encoded quality into Q'''
    return ord(qual) - 33 # converts integer to character according to ASCII table

def create_history(qualities):
    history = [0] * 50
    for qual in qualities:
        for phred in qual:
            q = phred33ToQ(phred)
            history[q] += 1
    return history

def max_poor_quality_sequencing_cycle(qualities):
    min_score = 123456789
    min_index = -1
    for i, qual in enumerate(qualities):
        score = sum(map(ord, qual))
        if min_score > score:
            min_score = score
            min_index = i
    return min_index


### Matching ###
def naive(p, t):
    """
    A naive method of finding all occurrences of `p` in `t`, where 
    `p` is the shorter sequence (pattern) and `t` is the longer (target) 
    reference sequence. 

    """
    occurrences = []
    for i in range(len(t) - len(p) + 1):  # loop over alignments
        match = True
        for j in range(len(p)):  # loop over characters
            if t[i+j] != p[j]:  # compare characters
                match = False
                break
        if match:
            occurrences.append(i)  # all chars matched; record
    return occurrences

def naive_2mm(p, t):
    occurrences = []
    for i in range(len(t) - len(p) + 1):  # loop over alignments
        mismatch = 0
        for j in range(len(p)):  # loop over characters
            if t[i+j] != p[j]:  # compare characters
                mismatch += 1
                if mismatch > 2:
                    break
        if mismatch <= 2:
            occurrences.append(i)  # all chars matched; record
    return occurrences

### General Utilities ### 

def download_to(link, output_dir=None, verbose=0): 
    # import glob
    import utils_sys as us

    if output_dir is None: 
        output_dir = os.path.join(os.getcwd(), 'data/demo')
    cmd = f"wget --directory-prefix={output_dir} --no-check {link}" # will create directory if it doesn't exist
    output = us.execute(cmd)

    # Get the latest file in the output directory 
    list_of_files = glob.glob(f'{output_dir}/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    if verbose: 
        print(f"(download to) wget output:\n{output}\n")
        print(f"(download to) Downloaded file:\n{latest_file}\n")
    output_path = os.path.join(output_dir, latest_file)
    return output_path

def savefig(plt, fpath, ext='tif', dpi=500, message='', verbose=True):
    """
    fpath: 
       name of output file
       full path to the output file

    Memo
    ----
    1. supported graphing format: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff 

    """
    import os

    # [todo] Configuration
    supported_formats = ['eps', 'jpeg', 'jpg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff', ]  

    output_dir = ''
    fname = 'generic-figure.%s' % ext
    if fpath and os.path.isdir(fpath): 
        # file name is not given, just an output dir
        outputdir = fpath
    else: 
        # not a directory, assuming it's a full path for which the file may not exist yet
        outputdir, fname = os.path.dirname(fpath), os.path.basename(fpath) 

        if fname: 
            # automatically infer ext
            if not ext: 
                # But this logic can be problematic because the file name itself may contain '.'
                fext = fname.split('.')
                if len(fext) >= 2: 
                    ext = fext[-1] # use the given file extension as the preferred extension
            else: 
                # Check if the input file name already includes the extension
                n = len(ext)
                if fname[-n:] == ext and fname[-(n+1)] == '.': 
                    # no-opt
                    pass
                else: 
                    fname = f"{fname}.{ext}"

    # supported graphing format: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    if not outputdir: outputdir = os.getcwd() # sys_config.read('DataExpRoot') # ./bulk_training/data-learner)
    assert os.path.exists(outputdir), "Invalid output path: %s" % outputdir

    ext_plot = ext  # eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    assert ext in supported_formats, "Unsupported graphic format: %s" % ext

    # fbase, fext = os.path.splitext(fname)
    # if len(fext) == 0: # file name does not contain extension
    #     fname = f"{fname}.{ext}"
    
    fpath = os.path.join(outputdir, fname)

    if verbose: print('(save_figure) Saving plot to:\n%s\n... description: %s' % (fpath, 'n/a' if not message else message))
    
    # NOTE: pylab leaves a generous, often undesirable, whitespace around the image. Remove it by setting bbox_inches to tight
    plt.savefig(fpath, bbox_inches='tight', dpi=dpi)   
    return

### Demo Functions ####

def demo_parsing_fasta(prefix='data/demo'):
    import utils_sys as us 
 
    link = "https://d28rh4a8wq0iu5.cloudfront.net/ads1/data/lambda_virus.fa"
    # prefix = os.path.join(os.getcwd(), prefix)
    cmd = f"wget --directory-prefix={prefix} --no-check {link}" # will create directory if it doesn't exist
    # output = us.execute(cmd)
    print("[demo] output:\n{output}\n")

    input_path = os.path.join(prefix, "lambda_virus.fa")

    genome = s = read_genome(input_path)
    print(s[:100])

    question1 = '''1. How many times does AGGT or its reverse complement (ACCT) occur in the lambda virus genome?
    E.g. if AGGT occurs 10 times and ACCT occurs 12 times, you should report 22.'''
    print(question1)   

    count = 0
    count += len(naive('AGGT', genome))
    count += len(naive(reverse_complement('AGGT'), genome))
    print(count)  
 
    ### Counting occurrences of sub-sequences
    question2 = '''2. How many times does TTAA or its reverse complement occur in the lambda virus genome?
    Hint: TTAA and its reverse complement are equal, so remember to not double count'''

    count = len(naive('TTAA', genome))
    print(count)

    ### First occurrence and last occurrence of a sub-sequence
    question3 = '''What is the offset of the rightmost occurrence of ACTAAGT or its reverse complement
    in the Lambda virus genome? E.g. if the rightmost occurrence of ACTAAGT is at offset 40 (0-based)
    and the rightmost occurrence of its reverse complement ACTTAGT is at offset 29, then report 29.'''
    print(question3)

    needle = 'ACTAAGT'
    offset1 = genome.rfind(needle)
    offset2 = genome.rfind(reverse_complement(needle))
    print('offset1: %d, offset2: %d' % (offset1, offset2))
    print(min(offset1, offset2))

    question4 = '''What is the offset of the leftmost occurrence of AGTCGA or its reverse complement
    in the Lambda virus genome?'''
    print(question4)

    needle = 'AGTCGA'
    offset1 = genome.find(needle)
    offset2 = genome.find(reverse_complement(needle))
    print('offset1: %d, offset2: %d' % (offset1, offset2))
    print(min(offset1, offset2))


    question5 = '''As we will discuss, sometimes we would like to find approximate matches for P in T.
That is, we want to find occurrences with one or more differences.

For Questions 5 and 6, make a new version of the naive function called naive_2mm
that allows up to 2 mismatches per occurrence. Unlike for the previous questions,
do not consider the reverse complement here. We're looking for approximate matches for P itself,
not its reverse complement.

For example, ACTTTA occurs twice in ACTTACTTGATAAAGT, once at offset 0 with 2 mismatches,
and once at offset 4 with 1 mismatch. So naive_2mm(’ACTTTA’,’ACTTACTTGATAAAGT’)
should return the list [0,4].

How many times does TTCAAGCC occur in the Lambda virus genome when allowing up to 2 mismatches?'''
    print(question5)

    s1, s2 = 'ACTTTA', 'ACTTACTTGATAAAGT'
    print(naive_2mm(s1, s2))
    print()

    s = 'TTCAAGCC'
    count = len(naive_2mm(s, genome))
    print(count); print()

    question6 = '''What is the offset of the leftmost occurrence of AGGAGGTT
in the Lambda virus genome when allowing up to 2 mismatches?'''
    print(question6)

    s = 'AGGAGGTT'
    offsets = naive_2mm(s, genome)
    print(offsets[0])

    ### Sequencing Quality 

    question7 = '''Finally, download and parse the provided FASTQ file containing real DNA sequencing reads
derived from a human:

https://d28rh4a8wq0iu5.cloudfront.net/ads1/data/ERR037900_1.first1000.fastq

Note that the file has many reads in it and you should examine all of them together when answering this question.
The reads are taken from this study:

Ajay, S. S., Parker, S. C., Abaan, H. O., Fajardo, K. V. F., & Margulies, E. H. (2011). Accurate

and comprehensive sequencing of personal genomes. Genome research, 21(9), 1498-1505.

This dataset has something wrong with it; one of the sequencing cycles is poor quality.

Report which sequencing cycle has the problem. Remember that a sequencing cycle corresponds
to a particular offset in all the reads. For example, if the leftmost read position seems
to have a problem consistently across reads, report 0. If the fourth position from the left has the problem,
report 3. Do whatever analysis you think is needed to identify the bad cycle.
It might help to review the "Analyzing reads by position" video.'''
    print(question7)

    link = "https://d28rh4a8wq0iu5.cloudfront.net/ads1/data/ERR037900_1.first1000.fastq"
    file_name = link.split("/")[-1]
    # prefix = os.path.join(os.getcwd(), "data/demo")
    file_path = os.path.join(prefix, file_name)
    if not os.path.exists(file_path): 
        file_path = download_to(link, verbose=1)
    else: 
        print(f"fastq file already exists at:\n{file_path}\n")
    print(f"> Downloaded fastq file:\n{file_path}\n")

    seq, qual = read_fastq(file_path)
    print(seq[:5])
    print(qual[:5])
    print(len(seq[0]))

    h = create_history(qual)
    print(h)

    plt.bar(range(len(h)), h)
    # plt.show()
    output_path = os.path.join(prefix, "quality_score.tif")
    savefig(plt, output_path, dpi=300, message='', verbose=True)

    offset = max_poor_quality_sequencing_cycle(qual)
    print("incorrect answer: %d" % offset)


    return

def demo_algorithm(): 

    s1, s2 = 'ACCAGTC', 'ACCATTG'
    print(longestCommonPrefix(s1, s2))

    s1, s2 = 'ACCA', 'ACCA'
    print(match(s1, s2))

    s = 'AGGCC'
    print(reverse_complement(s))

    s = 'abcdefghijklmnopqrstuvwxyzzzzabc'
    print(s)
    for i in range(len(s)):
        print(i % 10, end='')
    print(); 

    # s.rfind('tuv') # find the index of 'tuv'
    print(f"> {s}")
    print(f"> s.rfind('tuv'): {s.rfind('tuv')}" )
    print(f"> s.rfind('abc'): {s.rfind('abc')}") # find last occurrence
    print(f"> s.find('abc'): {s.find('abc')}")  # find first occurrence

    return

def demo_savefig(**kargs): 
    from utils_classifier import generate_imbalanced_data
    import seaborn as sns
    from pathlib import Path

    # X, y = generate_imbalanced_data(class_ratio=0.95, verbose=1)
    # print(f"> shape(X): {X.shape}, shape(y): {y.shape}")
    # print(collections.Counter(y))

    tips = sns.load_dataset("tips")
    print(tips.head())

    # Plot data
    sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time")

    #############################################################
    ext = 'tif'
    experiment = 'plotting_demo'
    data_dir = os.path.join(os.getcwd(), f"data/demo/{experiment}")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    output_dir = kargs.get("output_dir", data_dir)
    output_file = kargs.get("output_file", f"imbalanced_data.{ext}")
    # os.path.join(os.getcwd(), "result_expr_significance")
    output_path = kargs.get("output_path", os.path.join(output_dir, output_file))
    savefig(plt, output_path, ext=ext, dpi=300, message='', verbose=True)
    #############################################################

    return


def test(): 

    # ----- String manipulations ---- 
    # demo_algorithm()

    # ----- Parsing FASTA file ----- 
    # demo_parsing_fasta()

    # ----- Plotting Demo ----- 
    demo_savefig()

    return

if __name__ == "__main__":
    test()
