# meta-spliceai/
# ├── README.md
# ├── build/
# ├── demo/
# ├── environment.yml
# ├── environment_explicit.txt
# ├── output/
# ├── pyproject.toml
# ├── requirements.txt
# ├── requirements.txt.bk
# ├── setup.py
# ├── meta_spliceai/
# │   ├── __init__.py
# │   ├── __pycache__/
# │   ├── analyze_pub_results.py
# │   ├── create_query_uorfs.py
# │   ├── data_generation_workflow.py
# │   ├── data_pipeline.py
# │   ├── demo/
# │   ├── experiments/
# │   ├── extract_junctions.py
# │   ├── fetch_gene_syns.py
# │   ├── mllib/
# │   ├── optimization/
# │   ├── run_data_generation_workflow.py
# │   ├── run_test_retrieval.py
# │   ├── sequence_model/
# │   ├── sphere_pipeline/
# │   ├── splice_engine/
# │   │   ├── __init__.py
# │   │   ├── analysis_utils.py
# │   │   ├── analyzer.py
# │   │   ├── document_error_analysis_feature_set.py
# │   │   ├── evaluate_models.py
# │   │   ├── extract_genomic_features.py
# │   │   ├── label_splice_sites.py
# │   │   ├── model_evaluator.py
# │   │   ├── overlapping_gene_mapper.py
# │   │   ├── performance_analyzer.py
# │   │   ├── prepare_splice_site_dataset.py
# │   │   ├── run_spliceai_workflow.py
# │   │   ├── run_spliceai_workflow_v0.py
# │   │   ├── sequence_featurizer.py
# │   │   ├── splice_error_analyzer.py
# │   │   ├── train_fp_model.py
# │   │   ├── utils.py
# │   │   ├── utils_bio.py
# │   │   ├── utils_doc.py
# │   │   ├── utils_fs.py
# │   │   ├── utils_plot.py
# │   │   ├── visual_analyzer.py
# │   ├── system/
# │   ├── target_selection/
# │   ├── test.gtf
# │   ├── test_retrieval.py
# │   ├── tests/
# │   ├── transcript_features.csv
# │   ├── uORF_explorer/
# ├── tests/

from setuptools import setup, find_packages

install_requires = [
    # 'argcomplete==3.5.2',
    # 'argh==0.31.3',
    # 'gffutils==0.13',
    # 'h5py==3.12.1',
    # 'keras==3.7.0',
    # 'ml-dtypes==0.4.1',
    # 'optree==0.13.1',
    # 'polars==1.17.1',
    # 'protobuf==5.29.1',
    # 'pybedtools==0.10.0',
    # 'pyfaidx==0.8.1.3',
    # 'pyparsing==3.2.0',
    # 'requests==2.32.3',
    # 'rich==13.9.4',
    # 'simplejson==3.19.3',
    # 'six==1.17.0',
    # 'spliceai==1.3.1',
    # 'tensorboard==2.18.0',
    # 'tensorflow==2.18.0',
    # 'tqdm==4.67.1', 
    # 'ace-tool'
]

setup(
    name='meta-spliceai',  
    version='0.1.3',  
    description='Meta-learning framework for adaptive splice site prediction and novel isoform discovery',  # Short description
    author='Barnett Chiu',
    author_email='barnettchiu@gmail.com',
    url='https://github.com/pleiadian53/meta-spliceai',  # Create repo before first push
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),  # Automatically find packages in your directory
    install_requires=install_requires, # List of dependencies; use `pip install -r requirements.txt` to install all dependencies
    
    extras_require={
        'dev': ['pytest', ],
        # 'test': ['coverage', 'mock'],
    },
    # entry_points={
    #     'console_scripts': [
    #     ],
    # },

    include_package_data=True,
    package_data={
        'meta_spliceai': ['.png', '.webp', '.xlsx', '*.csv', '*.tsv', '*.md', '.pdf'],
    },    
)
