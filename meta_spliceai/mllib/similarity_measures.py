


###### Set similarity measures ###### 
# Regarding other set similarity metrics, there are several. Some of them are:
# Dice coefficient: It's similar to the Jaccard index but is defined as 
# 
#     2(A ^ B)
#    ----------
#    |A| + |B|
# 
# Hamming distance: It measures the minimum number of substitutions required to change one string into the other, or the minimum number of errors that could have transformed one string into the other.
# Tanimoto coefficient: This is essentially equivalent to the Jaccard coefficient for binary data.
# 
# Among these, the Jaccard index and Dice coefficient are particularly popular for set-based similarity computations.

### Jaccard Index

def compute_jaccard(set1, set2):
    """Compute the Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def pairwise_jaccard_similarity(adict):
    """Compute pairwise Jaccard similarity for all entries in a dictionary."""
    keys = list(adict.keys())
    result = {}
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            key1, key2 = keys[i], keys[j]
            jaccard_sim = compute_jaccard(set(adict[key1]), set(adict[key2]))
            result[(key1, key2)] = jaccard_sim
    return result

def test_pairwise_jaccard_similarity():
    # Test case 1
    adict1 = {'a': [1, 3, 5, 7], 'b': [2, 3, 4, 5], 'c': [0, 9, 2, 8, 7]}
    result1 = pairwise_jaccard_similarity(adict1)
    expected1 = {
        ('a', 'b'): 2/6,
        ('a', 'c'): 1/8,
        ('b', 'c'): 1/8
    }
    assert result1 == expected1, f"Expected {expected1}, but got {result1}"

    # Test case 2
    adict2 = {'x': [1, 2, 3], 'y': [3, 4, 5], 'z': [5, 6, 7], 'w': [1, 3, 5]}
    result2 = pairwise_jaccard_similarity(adict2)
    expected2 = {
        ('x', 'y'): 1/5,
        ('x', 'z'): 0/6,
        ('x', 'w'): 2/4,  # Corrected value
        ('y', 'z'): 1/5,
        ('y', 'w'): 2/4,  # Corrected value
        ('z', 'w'): 1/5
    }
    assert result2 == expected2, f"Expected {expected2}, but got {result2}"

    # Test case 3: Edge case with empty sets
    adict3 = {'p': [], 'q': [1, 2, 3]}
    result3 = pairwise_jaccard_similarity(adict3)
    expected3 = {
        ('p', 'q'): 0.0
    }
    assert result3 == expected3, f"Expected {expected3}, but got {result3}"

    print("All tests passed!")

    # --- Test ----

    adict1 = {'a': [1, 3, 5, 7], 'b': [2, 3, 4, 5], 'c': [0, 9, 2, 8, 7]}
    result1 = pairwise_jaccard_similarity(adict1)
    print(display_results_with_tabulate(result1, measure='Jaccard Index'))

    return 


### Dice coefficient

def compute_dice_coeff(set1, set2):
    """
    Compute the Dice coefficient between two sets.

    Args:
    - set1 (set): The first set.
    - set2 (set): The second set.

    Returns:
    - float: The Dice coefficient between the two sets.
    """
    intersection_size = len(set1.intersection(set2))
    return (2 * intersection_size) / (len(set1) + len(set2))


def pairwise_dice_coeff(input_dict):
    """
    Compute the pairwise Dice coefficient for all pairs of sets in the input dictionary.

    Args:
    - input_dict (dict): A dictionary where keys are labels and values are sets.

    Returns:
    - dict: A dictionary where keys are pairs of labels and values are the Dice coefficients
            between the sets corresponding to those labels.
    """
    pairwise_coefficients = {}
    keys = list(input_dict.keys())
    
    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):
            if i < j:
                coeff = compute_dice_coeff(input_dict[key1], input_dict[key2])
                pairwise_coefficients[(key1, key2)] = coeff
                
    return pairwise_coefficients

def test_pairwise_dice_coefficient(): 
    # Test cases for Dice coefficient functions

    test_case_1 = {
        'x': {1, 2, 3, 4},
        'y': {3, 4, 5, 6},
        'z': {5, 6, 7, 8}
    }

    test_case_2 = {
        'p': {10, 20, 30, 40},
        'q': {10, 20},
        'r': {30, 40}
    }

    test_case_3 = {
        'm': {1, 2, 3},
        'n': {4, 5, 6},
        'o': {7, 8, 9}
    }

    results_1 = pairwise_dice_coeff(test_case_1)
    results_2 = pairwise_dice_coeff(test_case_2)
    results_3 = pairwise_dice_coeff(test_case_3)

    print(display_results_with_tabulate(results_1))
    print(display_results_with_tabulate(results_2))
    print(display_results_with_tabulate(results_3))
    
    return

def display_results_with_tabulate(results_dict, measure='Dice Coefficient'):
    from tabulate import tabulate  # This import may not work here, but it should work in your local environment
    
    # Prepare the data for tabulation
    table_data = []
    keys = list(results_dict.keys())
    
    # Add headers
    headers = ["Pair", measure]
    
    # Populate the table data
    for pair, value in results_dict.items():
        table_data.append([pair, value])
    
    # Use tabulate to display the table
    table = tabulate(table_data, headers=headers, tablefmt='grid')
    
    return table


def test(): 

    test_pairwise_jaccard_similarity()

    # test_pairwise_dice_coefficient()

    return

if __name__ == "__main__": 
    test()
