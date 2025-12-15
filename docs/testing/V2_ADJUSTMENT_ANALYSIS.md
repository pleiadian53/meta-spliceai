# V2 Correlated Probability Vector Adjustment - Analysis

## Date: 2025-10-31

## Problem Identified

The "unified" adjustment method in v2 is too conservative:
- It only adjusts positions where the base model predicts high scores (>0.1)
- For VCAM1: Only 9 donor and 9 acceptor positions were adjusted out of 19,304 total
- This leaves most positions unadjusted, leading to poor alignment

## Root Cause

The unified method determines the "dominant type" BEFORE adjustment and only adjusts those positions. But we need to apply the adjustment to ALL positions because:
1. We're trying to align the coordinate system globally, not selectively
2. A position might have a low donor score at position X, but a high donor score at position X+2 (after adjustment)

## Correct Approach

We need to create **separate views** for evaluating different splice types:

### Donor Evaluation
When checking if a position is a donor:
1. Use the "donor view" where ALL scores are shifted by the donor adjustment
2. This ensures the donor score at position X reflects the signal from position X+2 (for +2 adjustment)
3. The acceptor and neither scores at position X also come from position X+2, maintaining correlation

### Acceptor Evaluation  
When checking if a position is an acceptor:
1. Use the "acceptor view" where ALL scores are shifted by the acceptor adjustment
2. This ensures the acceptor score at position X reflects the signal from the correct offset position

## Implementation Plan

Update `score_adjustment_v2.py` to:
1. Always create separate donor and acceptor views (shift ALL positions, not just high-scoring ones)
2. Return a dictionary with both views
3. Update the evaluation code to use the appropriate view for each splice type

## Key Insight

The three scores (donor, acceptor, neither) at each position are correlated and must move together as a unit. But we need DIFFERENT views for evaluating DIFFERENT splice types, because donors and acceptors have different coordinate adjustments.

This is analogous to having two different "lenses" through which to view the data:
- Donor lens: optimized for finding donors
- Acceptor lens: optimized for finding acceptors

