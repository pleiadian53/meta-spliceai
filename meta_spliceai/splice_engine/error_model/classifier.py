"""
Error classifier module for training and evaluating error models.

This module provides a clean interface to the error classifier functionality,
bridging the existing error_classifier.py implementation with the new package structure.
"""

import os
import sys
import inspect
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

# Import from existing modules
# Design Pattern: The underscore prefix (_train_error_classifier) indicates this is an 
# implementation detail. This follows the facade pattern where we import the original 
# implementation but expose it through a cleaner, more object-oriented API (ErrorClassifier).
# This approach allows us to refactor the internals without breaking external API compatibility.
from ..model_training.error_classifier import (
    train_error_classifier as _train_error_classifier,
)


from ..splice_error_analyzer import ErrorAnalyzer
from ..model_evaluator import ModelEvaluationFileHandler
# Removed import for get_importance_df as it doesn't exist as a top-level function

class ErrorClassifier:
    """
    Class for training and analyzing error classifiers.
    
    This class wraps the functionality in model_training.error_classifier with
    a more object-oriented interface.
    """
    
    def __init__(
        self,
        error_label: str,
        correct_label: str,
        splice_type: str,
        experiment: str = "error_analysis",
        model_type: str = "xgboost",
        output_dir: Optional[str] = None,
        verbose: int = 1,
        input_dir: str = None,
        separator: str = None
    ):
        """
        Initialize an ErrorClassifier instance.
        
        Parameters
        ----------
        error_label : str
            Label for error class (e.g., "FP", "FN")
        correct_label : str
            Label for correct prediction class (e.g., "TP", "TN")
        splice_type : str
            Type of splice site ("donor", "acceptor", or "any")
        experiment : str, default="error_analysis"
            Experiment name for organizing outputs
        output_dir : str, optional
            Custom output directory. If None, uses the analyzer's default directory.
        verbose : int, default=1
            Verbosity level
        input_dir : str, optional
            Custom input directory for ModelEvaluationFileHandler
        separator : str, optional
            Custom separator for ModelEvaluationFileHandler
        """
        self.error_label = error_label
        self.correct_label = correct_label
        self.splice_type = splice_type
        self.experiment = experiment
        self.verbose = verbose
        
        # Set up analyzer for data access
        self.analyzer = ErrorAnalyzer(experiment=experiment, model_type=model_type.lower())
        
        # Handle output directory
        if output_dir is None:
            self.output_dir = self.analyzer.set_analysis_output_dir(
                error_label=error_label, 
                correct_label=correct_label, 
                splice_type=splice_type  # if donor or acceptor, then will subset by splice type
            )
        else:
            self.output_dir = output_dir

        # Handle input directory 
        if input_dir is None:
            self.input_dir = self.analyzer.eval_dir
        else:
            self.input_dir = input_dir
            
        # Default separator if none provided
        if separator is None:
            self.separator = "\t"
        else:
            self.separator = separator
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model and results
        self.model = None
        self.results = None
        self.data = None
        
    def load_data(
        self,
        custom_data: Optional[pd.DataFrame] = None,
        sample_ratio: Optional[float] = None,
        max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load error data from analyzer or use custom data.
        
        Parameters
        ----------
        custom_data : pd.DataFrame, optional
            Custom dataset to use instead of loading from analyzer
        sample_ratio : float, optional
            Ratio of samples to use (for faster development/testing)
        max_samples : int, optional
            Maximum number of samples to use
            
        Returns
        -------
        pd.DataFrame
            Dataset containing features and labels
        """
        if custom_data is not None:
            self.data = custom_data
        else:
            # Use ErrorAnalyzer to load featurized dataset
            self.data = self.analyzer.load_featurized_dataset()

            # Or, use ModelEvaluationFileHandler to load featurized dataset
            # mefd = ModelEvaluationFileHandler(self.input_dir, separator=self.separator)
            # self.data = mefd.load_featurized_dataset(
            #     aggregated=True, 
            #     error_label=self.error_label, 
            #     correct_label=self.correct_label,
            #     splice_type=self.splice_type
            # )
        
        # Apply sampling if requested
        if sample_ratio is not None or max_samples is not None:
            from .workflow import apply_stratified_sampling
            
            self.data = apply_stratified_sampling(
                self.data,
                sampling_ratio=sample_ratio,
                max_samples=max_samples,
                verbose=self.verbose
            )
            
        return self.data
    
    def train(self, 
        data=None, 
        importance_methods=["shap", "xgboost", "mutual_info"], 
        top_k=20, 
        use_advanced_feature_plots=True, 
        feature_plot_type="box", 
        n_splits=5,
        **kwargs):
        """
        Train the error classifier.
        
        Parameters
        ----------
        data : DataFrame, optional
            Data to use for training. If None, uses self.data.
        importance_methods : list, optional
            List of feature importance methods to include.
        top_k : int, default=20
            Number of top features to display in visualizations
        use_advanced_feature_plots : bool, default=True
            Whether to use advanced feature distribution plots
        feature_plot_type : str, default="box"
            Type of plot for feature distributions (box, violin, etc.)
        **kwargs : dict
            Additional keyword arguments passed to train_error_classifier
            
        Returns
        -------
        tuple
            (model, results)
        """
        # Ensure we have data
        if data is not None:
            self.data = data
        elif self.data is None:
            self.load_data()
            
        # Use the parameter namespace pattern to avoid duplicate parameters
        import inspect
        
        # Get the signature of _train_error_classifier
        train_sig = inspect.signature(_train_error_classifier)
        train_param_names = set(train_sig.parameters.keys())
        
        # Create a parameter dictionary with explicit parameters
        train_params = {
            'error_label': self.error_label,
            'correct_label': self.correct_label,
            'splice_type': self.splice_type,
            'test_data': self.data,
            'output_dir': self.output_dir,
            'importance_methods': importance_methods,
            'top_k': top_k,
            'use_advanced_feature_plots': use_advanced_feature_plots,
            'feature_plot_type': feature_plot_type,
            'n_splits': n_splits,
            'verbose': self.verbose
        }
        
        # Add additional parameters from kwargs, avoiding duplicates
        for param_name, param_value in kwargs.items():
            if param_name in train_param_names and param_name not in train_params:
                train_params[param_name] = param_value
        
        # Train the classifier and get results
        self.model, self.results = _train_error_classifier(**train_params)
        
        # Create a status file
        status_file = os.path.join(
            self.output_dir, 
            f"{self.splice_type}_{self.error_label}_vs_{self.correct_label}_done.txt"
        )
        
        with open(status_file, 'w') as f:
            f.write(f"Completed processing {self.splice_type}: {self.error_label} vs {self.correct_label}")
        
        return self.model, self.results
    
    def get_top_features(self, method: str = "shap", k: int = 20) -> List[str]:
        """
        Get the top k features according to the specified importance method.
        
        Parameters
        ----------
        method : str, default="shap"
            Feature importance method to use (shap, xgboost, mutual_info, hypothesis)
        k : int, default=20
            Number of top features to return
            
        Returns
        -------
        list
            List of top feature names
        """
        if self.results is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        # Get importance dataframe based on method
        if method == "shap" and "importance_df_shap" in self.results:
            importance_df = self.results["importance_df_shap"]
        elif method == "xgboost" and "importance_df_xgb" in self.results:
            importance_df = self.results["importance_df_xgb"]
        elif method in ["mutual_info", "hypothesis"] and "importance_df_stats" in self.results:
            importance_df = self.results["importance_df_stats"].query(f"method == '{method}'")
        else:
            # If requested method not available, try model's feature_importance
            if self.model is not None and hasattr(self.model, "feature_importances_"):
                features = self.data.drop(columns=["label"]).columns.tolist()
                importance = self.model.feature_importances_
                importance_df = pd.DataFrame({
                    "feature": features,
                    "importance": importance
                }).sort_values("importance", ascending=False)
            else:
                raise ValueError(f"Feature importance method '{method}' not available in results.")
                
        # Return top k features
        return importance_df["feature"].head(k).tolist()
    
    def compare_feature_importance(
        self,
        methods: List[str] = ["shap", "xgboost", "mutual_info"],
        top_k: int = 15,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare feature importance rankings from different methods.
        
        Parameters
        ----------
        methods : list, default=["shap", "xgboost", "mutual_info"]
            Feature importance methods to compare
        top_k : int, default=15
            Number of top features to include in comparison
        output_file : str, optional
            Custom filename for saving the comparison plot
            
        Returns
        -------
        pd.DataFrame
            DataFrame with feature importance rankings from different methods
        """
        if self.results is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        from ..model_training.feature_importance import compare_feature_importance_ranks
        
        if output_file is None:
            output_file = f"{self.splice_type}_{self.error_label}_vs_{self.correct_label}_feature_importance_comparison.pdf"
            
        output_path = os.path.join(self.output_dir, output_file)
        
        # Get available importance DataFrames
        importance_dfs = {}
        
        if "shap" in methods and "importance_df_shap" in self.results:
            importance_dfs["SHAP"] = self.results["importance_df_shap"]
            
        if "xgboost" in methods and "importance_df_xgb" in self.results:
            importance_dfs["XGBoost"] = self.results["importance_df_xgb"]
            
        if "importance_df_stats" in self.results:
            stats_df = self.results["importance_df_stats"]
            
            if "mutual_info" in methods:
                mi_df = stats_df.query("method == 'mutual_info'").copy()
                if not mi_df.empty:
                    importance_dfs["Mutual Info"] = mi_df
                    
            if "hypothesis" in methods:
                hyp_df = stats_df.query("method == 'hypothesis'").copy()
                if not hyp_df.empty:
                    importance_dfs["Hypothesis"] = hyp_df
        
        # Ensure we have at least two methods for comparison
        if len(importance_dfs) < 2:
            raise ValueError("Need at least two feature importance methods for comparison.")
            
        # Call the comparison function
        comparison_result = compare_feature_importance_ranks(
            importance_dfs,
            output_file=output_path,
            top_k=top_k,
            verbose=self.verbose
        )
        
        return comparison_result


# Function aliases for direct API compatibility
# Design Pattern: Module-level function alias that maintains backward compatibility
# while encouraging migration to the OOP interface (ErrorClassifier class).
# This allows existing code to continue using the functional API (train(...)) 
# while new code can use the more maintainable object-oriented approach.
# The underscore prefix in the imported name (_train_error_classifier) and 
# the non-prefixed alias (train) clearly distinguish between implementation and public API.
train = _train_error_classifier

# Direct import 
# from meta_spliceai.splice_engine.error_model.classifier import train

# Module level access as a Public API Component
# import meta_spliceai.splice_engine.error_model.classifier as error_model
# model, results = error_model.train(...)  # Function call at module level

# Object-oriented interface
# error_model = ErrorClassifier(
#     error_label="FP",
#     correct_label="TP",
#     splice_type="donor"
# )
# model, results = error_model.train()
