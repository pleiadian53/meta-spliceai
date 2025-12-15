# Documentation: `position_centric_data_repr_diagram.py`

This document provides detailed information for the script `position_centric_data_repr_diagram.py`.

---

## Purpose

This script generates a diagram visualizing the shared "position-centric" data representation used in both the error modeling and meta-model stages of MetaSpliceAI. This visualization is crucial for understanding how sequence context and probability vectors are integrated in the project.

The output diagram is saved as `position_centric_data_representation.png` in the `results/plots/` directory.

### Generated Diagram

![Position-Centric Data Representation](results/plots/position_centric_data_representation.png)

## Requirements

This script requires the `graphviz` library. It is essential to install both the Python package and the underlying system binaries for it to function correctly.

### Installation

The recommended way to install the dependency within the `surveyor` conda environment is from the `conda-forge` channel:

```bash
conda install -c conda-forge python-graphviz
```

This single command installs both the Python wrapper and the necessary Graphviz system executables (like the `dot` engine), which is required for rendering the diagram.

## Troubleshooting

Here are common issues encountered when running this script:

- **`FileNotFoundError: [Errno 2] No such file or directory: 'dot'`**
  - **Cause**: This error occurs if the Graphviz system executables are not found in your system's PATH. It typically means you only installed the Python wrapper (e.g., via `pip install graphviz`) without installing the core Graphviz software.
  - **Solution**: The `conda install` command listed above is the most reliable way to fix this, as it ensures both components are installed correctly.

- **Permission Errors when saving the file**
  - **Cause**: The script saves its output to the `results/` directory within the project root. A permission error indicates that your user does not have write access to this location.
  - **Solution**: Ensure you have the necessary write permissions for the `results/` directory in your project root.
