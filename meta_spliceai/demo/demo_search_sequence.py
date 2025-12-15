import re


def demo_search_sequence():
    """

    This function demonstrates how to search for a specific sequence in a gene sequence.
    Expected output:

    Match: CAG, Start: 3, End: 6 (exclusive)
    Match: TAG, Start: 6, End: 9
    Match: AAG, Start: 9, End: 12
    """
    gene_seq = "ATGCAGTAGAAGCTG"
    for match in re.finditer(r'CAG|TAG|AAG', gene_seq):  # Adjust regex for other donor/acceptor sites
        match_start = match.start()
        match_end = match.end()
        print(f"Match: {match.group()}, Start: {match_start}, End: {match_end}")

def test(): 
    demo_search_sequence()


if __name__ == "__main__": 
    test()