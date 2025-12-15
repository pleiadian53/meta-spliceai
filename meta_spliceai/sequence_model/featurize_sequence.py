import os, sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

# import meta_spliceai.sequence_model.data_model as dm
from meta_spliceai.sequence_model.data_model import Sequence, SequenceMarkers


# Define the function to compute k-mer counts
def get_kmer_counts(sequence, k):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k), lowercase=False)
    kmers = vectorizer.fit_transform([sequence])
    kmer_counts = kmers.toarray().sum(axis=0)
    kmer_names = vectorizer.get_feature_names_out()
    return dict(zip(kmer_names, kmer_counts))

# Define the function to compute GC content
def get_gc_content(sequence, upper=True):
    if upper: 
        return (sequence.count('G') + sequence.count('C')) / len(sequence)
    return (sequence.count('g') + sequence.count('c')) / len(sequence)

# Define the function to compute sequence length
def get_sequence_length(sequence):
    return len(sequence)

# Define the function to compute marker counts
def get_marker_counts(marker, markers):
    marker_counts = {marker_type: marker.count(str(symbol)) for marker_type, symbol in markers.items()}
    return marker_counts

# Define the function to compute transition counts between markers
def get_transition_counts(marker):
    transitions = {}
    for i in range(len(marker) - 1):
        transition = marker[i:i+2]
        if transition not in transitions:
            transitions[transition] = 0
        transitions[transition] += 1
    return transitions

# Define the function to compute complexity of sequence
def get_sequence_complexity(sequence):
    frequency_list = [sequence.count(nucleotide) for nucleotide in set(sequence)]
    sequence_complexity = pd.Series(frequency_list).apply(lambda x: -x/len(sequence) * np.log2(x/len(sequence))).sum()
    return sequence_complexity