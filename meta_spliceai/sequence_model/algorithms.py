import os, sys
import re


def is_overlapping(a, b):
    x1, x2 = a  # ---x1----x2
    y1, y2 = b  # -------y1----y2
    return max(x1,y1) <= min(x2,y2)

def has_overlaps(arr):
    # Sort intervals in increasing order of
    # start time
    n = len(arr)
    arr.sort()

    # In the sorted array, if end time of an
    # interval is not more than that of
    # end of previous interval, then there
    # is an overlap
    for i in range(1, n):
        if is_overlapping(arr[i-1], arr[i]): 
            return True
        # if (arr[i][0] <= arr[i - 1][1]):
        #     return True
    # If we reach here, then no overlap
    return False

def test(): 

    arr1 = [[1, 3], [1, 7], [4, 8 ], [2, 5]] # yes
    arr2 = [[1, 3], [7, 9], [4, 6], [10, 13]] # no

    for arr in [arr1, arr2, ]:
        print(f"> {arr[:5]} has overlaps? {has_overlaps(arr)}")


if __name__ == "__main__": 
    test()