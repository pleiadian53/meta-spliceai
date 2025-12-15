# Matplotlib Best Practices in MetaSpliceAI

## Handling Figure Cleanup

### Problem: Too Many Open Figures Warning

When running intensive visualization pipelines like `test_real_data_workflow.py`, you might encounter the following warning:

```
RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface 
(`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory.
(To control this warning, see the rcParam `figure.max_open_warning`). 
Consider using `matplotlib.pyplot.close()`.
```

This warning occurs when multiple matplotlib figures are created but not properly closed, which can lead to:

1. Excessive memory usage
2. Potential memory leaks in long-running processes
3. Degraded performance

### Solution: Always Close Figures After Saving

To prevent this issue, always add `plt.close()` immediately after saving a figure with `plt.savefig()`:

```python
# Create and save a figure
plt.figure(figsize=(10, 8))
# ... plotting code ...
plt.savefig(output_path, dpi=300)
plt.close()  # Always close the figure after saving
```

### Where This Pattern Is Applied in MetaSpliceAI

The following components implement proper figure cleanup:

- `plot_feature_importance()` in `analysis_utils.py`
- `plot_cv_roc_curve()` and `plot_cv_pr_curve()` in `performance_analyzer.py`
- SHAP importance plotting in `xgboost_trainer.py`

### Additional Tips

1. **Use Context Managers**: For more complex code, consider using context managers:
   ```python
   with plt.figure(figsize=(10, 8)) as fig:
       # plotting code
       plt.savefig(output_path)
       # No need to explicitly call plt.close()
   ```

2. **Avoid Interactive Mode**: In scripts that generate many plots, avoid using `plt.ion()` (interactive mode), as it can complicate figure cleanup.

3. **Increase Warning Threshold**: If you're intentionally generating many figures, you can increase the warning threshold:
   ```python
   import matplotlib as mpl
   mpl.rcParams['figure.max_open_warning'] = 50  # Default is 20
   ```

## References

- [Matplotlib Documentation on Memory Management](https://matplotlib.org/stable/tutorials/intermediate/artists.html#memory-management-in-mpl)
- [Controlling Figure Creation](https://matplotlib.org/stable/users/explain/figure/pyplot.html)
