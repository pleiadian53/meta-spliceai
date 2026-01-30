import polars as pl
import inspect

# Print Polars version
print(f"Polars version: {pl.__version__}")

# Check scan_csv parameters
print("\nscan_csv parameters:")
print(inspect.signature(pl.scan_csv))
