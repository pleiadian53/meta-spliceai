#!/usr/bin/env python
"""
Test script to demonstrate how to use the schema_mismatch utility function
for diagnosing stack failures between donor and acceptor dataframes.
"""

import os
import sys
import polars as pl
import pandas as pd
import argparse
from pathlib import Path

# Import our schema utilities
from meta_spliceai.splice_engine.meta_models.core.schema_utils import analyze_schema_mismatch, ensure_schema, DEFAULT_POSITION_SCHEMA

def analyze_and_fix_dataframes(df1_path, df2_path, df1_name="donor", df2_name="acceptor"):
    """
    Demonstrate analyzing schema mismatches between two dataframes and fixing them.
    """
    # Load the dataframes
    print(f"Loading {df1_name} dataframe from {df1_path}")
    df1 = pl.read_csv(df1_path, separator='\t')
    
    print(f"Loading {df2_name} dataframe from {df2_path}")
    df2 = pl.read_csv(df2_path, separator='\t')
    
    print(f"\n{df1_name} dataframe has {df1.shape[0]} rows and {len(df1.columns)} columns")
    print(f"{df2_name} dataframe has {df2.shape[0]} rows and {len(df2.columns)} columns")
    
    # Print the schema for both dataframes
    print(f"\n{df1_name} schema:")
    for col, dtype in zip(df1.columns, df1.dtypes):
        print(f"  {col}: {dtype}")
    
    print(f"\n{df2_name} schema:")
    for col, dtype in zip(df2.columns, df2.dtypes):
        print(f"  {col}: {dtype}")
    
    # Try to stack them and see if it works
    print("\nAttempting to stack dataframes...")
    try:
        stacked = df1.vstack(df2)
        print(f"Success! Stacked dataframe has {stacked.shape[0]} rows")
        return None, None, stacked  # No mismatch, stacking worked
    except Exception as e:
        print(f"Stack failed with error: {e}")
        
        # Analyze the schema mismatch
        print("\nAnalyzing schema mismatch...")
        mismatch = analyze_schema_mismatch(df1, df2, df1_name, df2_name)
        
        # Print the analysis results
        print(f"\nColumns only in {df1_name}: {mismatch['columns_only_in_df1']}")
        print(f"Columns only in {df2_name}: {mismatch['columns_only_in_df2']}")
        print(f"Type mismatches: {mismatch['type_mismatches']}")
        
        print("\nDetailed mismatch information:")
        for detail in mismatch['mismatches_detail']:
            print(f"  Column: {detail['column']}")
            print(f"    {df1_name} type: {detail[f'{df1_name}_type']}, {df2_name} type: {detail[f'{df2_name}_type']}")
            print(f"    Is null issue: {detail['is_null_issue']}, Null-string issue: {detail['null_string_issue']}")
            print(f"    {df1_name} sample: {detail[f'{df1_name}_sample']}")
            print(f"    {df2_name} sample: {detail[f'{df2_name}_sample']}")
            print(f"    All null in {df1_name}: {detail[f'{df1_name}_all_null']}, All null in {df2_name}: {detail[f'{df2_name}_all_null']}")
        
        print(f"\nRecommendation: {mismatch['recommendation']}")
        
        # Try to fix the issues
        if mismatch['fixable']:
            print("\nAttempting to fix schema issues...")
            
            # Create a combined schema that handles all issues
            combined_schema = {}
            
            # Start with all columns from both dataframes
            all_columns = set(df1.columns).union(set(df2.columns))
            
            # For each column, determine the appropriate type
            for col in all_columns:
                # Check if this is a type mismatch column
                is_mismatch = col in mismatch['type_mismatches']
                
                if is_mismatch:
                    # Find the detail for this column
                    detail = next((d for d in mismatch['mismatches_detail'] if d['column'] == col), None)
                    
                    if detail and detail['null_string_issue']:
                        # For null vs string issues, use string type
                        combined_schema[col] = pl.Utf8
                    else:
                        # For other mismatches, prefer non-null type
                        if col in df1.columns and col in df2.columns:
                            dtype1 = df1.select(pl.col(col)).dtypes[0]
                            dtype2 = df2.select(pl.col(col)).dtypes[0]
                            
                            # Choose non-null type if available
                            if "null" in str(dtype1).lower() and "null" not in str(dtype2).lower():
                                combined_schema[col] = dtype2
                            elif "null" in str(dtype2).lower() and "null" not in str(dtype1).lower():
                                combined_schema[col] = dtype1
                            else:
                                # Default to Float64 for numeric and Utf8 for others
                                if "float" in str(dtype1).lower() or "float" in str(dtype2).lower():
                                    combined_schema[col] = pl.Float64
                                elif "int" in str(dtype1).lower() or "int" in str(dtype2).lower():
                                    combined_schema[col] = pl.Int64
                                else:
                                    combined_schema[col] = pl.Utf8
                        elif col in df1.columns:
                            combined_schema[col] = df1.schema[col]
                        else:
                            combined_schema[col] = df2.schema[col]
                else:
                    # Not a mismatch, use existing type
                    if col in df1.columns:
                        combined_schema[col] = df1.schema[col]
                    else:
                        combined_schema[col] = df2.schema[col]
            
            # Apply the combined schema to both dataframes
            print(f"Created combined schema with {len(combined_schema)} columns")
            df1_fixed = ensure_schema(df1, combined_schema)
            df2_fixed = ensure_schema(df2, combined_schema)
            
            # Try stacking again
            try:
                stacked = df1_fixed.vstack(df2_fixed)
                print(f"Success! Fixed and stacked dataframe has {stacked.shape[0]} rows")
                return df1_fixed, df2_fixed, stacked
            except Exception as e:
                print(f"Stack still failed after fixing: {e}")
                return df1_fixed, df2_fixed, None
        else:
            print("Schema issues are complex and may not be automatically fixable.")
            return None, None, None

def main():
    """
    Main function to test schema analysis with command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Analyze schema mismatches between two dataframes")
    parser.add_argument("--df1", type=str, required=True, help="Path to first dataframe CSV/TSV")
    parser.add_argument("--df2", type=str, required=True, help="Path to second dataframe CSV/TSV")
    parser.add_argument("--name1", type=str, default="DataFrame1", help="Name for first dataframe")
    parser.add_argument("--name2", type=str, default="DataFrame2", help="Name for second dataframe")
    parser.add_argument("--output", type=str, help="Output path for fixed stacked dataframe")
    
    args = parser.parse_args()
    
    # Run the analysis
    df1_fixed, df2_fixed, stacked = analyze_and_fix_dataframes(
        args.df1, args.df2, args.name1, args.name2
    )
    
    # Save output if requested and available
    if args.output and stacked is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving stacked dataframe to {out_path}")
        stacked.write_csv(str(out_path), separator='\t')
        print("Done!")

if __name__ == "__main__":
    main()
