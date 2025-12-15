from setuptools import setup, find_packages

setup(
    name="meta_spliceai_foundation_model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.8.0",
        "biopython>=1.79",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "pyfaidx>=0.6.4",
        "pybedtools>=0.8.2",
        "tqdm>=4.62.0",
        "shap>=0.40.0",
        "logomaker>=0.8"
    ],
    description="Deep learning foundation model for splice site prediction",
    author="MetaSpliceAI Team",
    python_requires=">=3.8",
)
