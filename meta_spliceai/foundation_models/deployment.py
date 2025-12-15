"""
Deployment utilities for splice site prediction model.

This module handles the packaging and deployment of trained models
for inference within MetaSpliceAI workflows.
"""

import os
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import time

logger = logging.getLogger(__name__)

class SplicePredictorModel:
    """Wrapper for deploying splice site prediction models."""
    
    def __init__(self, model=None, model_path=None, context_length=10000, 
                 threshold=0.5, batch_size=32, metadata=None):
        """
        Initialize the predictor model.
        
        Args:
            model: Trained model (optional)
            model_path (str): Path to saved model (optional)
            context_length (int): Context length used for prediction
            threshold (float): Classification threshold
            batch_size (int): Batch size for inference
            metadata (dict): Model metadata
        """
        self.context_length = context_length
        self.threshold = threshold
        self.batch_size = batch_size
        self.metadata = metadata or {}
        
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.load_model(model_path)
        else:
            logger.warning("No model provided. Please load a model before prediction.")
            self.model = None
    
    def load_model(self, model_path):
        """
        Load a saved model.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {model_path}")
        try:
            self.model = tf.keras.models.load_model(model_path)
            
            # Load metadata if available
            metadata_path = os.path.join(os.path.dirname(model_path), "model_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    
            return self.model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def save_model(self, save_dir, model_name="splice_predictor"):
        """
        Save the model and metadata.
        
        Args:
            save_dir (str): Directory to save model
            model_name (str): Name for the saved model
            
        Returns:
            str: Path to saved model
        """
        if self.model is None:
            logger.error("No model to save")
            return None
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, f"{model_name}.h5")
        logger.info(f"Saving model to {model_path}")
        self.model.save(model_path)
        
        # Add timestamp to metadata
        self.metadata["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.metadata["model_name"] = model_name
        
        # Save metadata
        metadata_path = os.path.join(save_dir, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
        return model_path
    
    def one_hot_encode(self, sequence):
        """
        One-hot encode a DNA sequence.
        
        Args:
            sequence (str): DNA sequence
            
        Returns:
            np.ndarray: One-hot encoded sequence
        """
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        seq_length = len(sequence)
        
        # Initialize encoded array
        encoded = np.zeros((seq_length, 5), dtype=np.float32)
        
        # Fill in encoded values
        for i, nucleotide in enumerate(sequence):
            if nucleotide in mapping:
                encoded[i, mapping[nucleotide]] = 1.0
            else:
                # For unknown nucleotides, use 'N' encoding
                encoded[i, 4] = 1.0
                
        return encoded
    
    def predict_single_sequence(self, sequence):
        """
        Predict splice sites for a single sequence.
        
        Args:
            sequence (str): DNA sequence
            
        Returns:
            float: Probability of being a splice site
        """
        if self.model is None:
            logger.error("No model loaded for prediction")
            return None
            
        # Ensure sequence is the right length
        if len(sequence) != self.context_length:
            logger.warning(f"Sequence length {len(sequence)} doesn't match model context length {self.context_length}")
            # Pad or truncate as needed
            if len(sequence) < self.context_length:
                sequence = sequence + 'N' * (self.context_length - len(sequence))
            else:
                sequence = sequence[:self.context_length]
        
        # One-hot encode the sequence
        encoded = self.one_hot_encode(sequence)
        
        # Make prediction
        prediction = self.model.predict(np.expand_dims(encoded, axis=0), verbose=0)[0][0]
        
        return float(prediction)
    
    def predict_batch(self, sequences, return_proba=True):
        """
        Predict splice sites for a batch of sequences.
        
        Args:
            sequences (list): List of DNA sequences
            return_proba (bool): If True, return probabilities; otherwise, return binary predictions
            
        Returns:
            np.ndarray: Predicted probabilities or binary predictions
        """
        if self.model is None:
            logger.error("No model loaded for prediction")
            return None
            
        # Prepare sequences
        processed_sequences = []
        for sequence in sequences:
            # Ensure sequence is the right length
            if len(sequence) != self.context_length:
                if len(sequence) < self.context_length:
                    sequence = sequence + 'N' * (self.context_length - len(sequence))
                else:
                    sequence = sequence[:self.context_length]
            
            # One-hot encode the sequence
            encoded = self.one_hot_encode(sequence)
            processed_sequences.append(encoded)
            
        # Stack sequences
        X = np.stack(processed_sequences, axis=0)
        
        # Make predictions
        probabilities = self.model.predict(X, batch_size=self.batch_size, verbose=0).flatten()
        
        if return_proba:
            return probabilities
        else:
            return (probabilities >= self.threshold).astype(int)
    
    def predict_from_fasta(self, fasta_file, window_size=None, stride=None):
        """
        Predict splice sites for sequences in a FASTA file.
        
        Args:
            fasta_file (str): Path to FASTA file
            window_size (int): Size of sliding window (default: context_length)
            stride (int): Stride for sliding window
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        from Bio import SeqIO
        
        if window_size is None:
            window_size = self.context_length
            
        if stride is None:
            stride = window_size // 2
            
        results = []
        
        # Process each sequence in the FASTA file
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq_id = record.id
            sequence = str(record.seq)
            
            # Skip if sequence is too short
            if len(sequence) < window_size:
                logger.warning(f"Sequence {seq_id} is shorter than window size, skipping")
                continue
                
            # Predict splice sites using sliding window
            for i in range(0, len(sequence) - window_size + 1, stride):
                window = sequence[i:i+window_size]
                prob = self.predict_single_sequence(window)
                
                results.append({
                    'sequence_id': seq_id,
                    'start': i,
                    'end': i + window_size,
                    'probability': prob,
                    'is_splice_site': prob >= self.threshold
                })
                
        # Convert to DataFrame
        df = pd.DataFrame(results)
        return df
    
    def export_for_surveyor(self, predictions_df, output_file):
        """
        Export predictions in a format compatible with MetaSpliceAI.
        
        Args:
            predictions_df (pd.DataFrame): DataFrame with predictions
            output_file (str): Path to output file
            
        Returns:
            str: Path to exported file
        """
        # Format for MetaSpliceAI
        formatted_df = predictions_df.copy()
        
        # Add required columns if not present
        if 'chromosome' not in formatted_df.columns:
            formatted_df['chromosome'] = formatted_df['sequence_id']
            
        if 'position' not in formatted_df.columns:
            # Use middle of the window as the position
            formatted_df['position'] = formatted_df['start'] + (formatted_df['end'] - formatted_df['start']) // 2
            
        if 'strand' not in formatted_df.columns:
            formatted_df['strand'] = '+'  # Default to positive strand
            
        # Rename columns to match MetaSpliceAI format
        formatted_df = formatted_df.rename(columns={
            'probability': 'splice_probability',
            'is_splice_site': 'predicted_splice'
        })
        
        # Save to file
        formatted_df.to_csv(output_file, index=False)
        logger.info(f"Exported predictions to {output_file}")
        
        return output_file
    
    def integrate_with_surveyor(self, input_data, output_path=None):
        """
        Integrate with MetaSpliceAI by processing input data and returning predictions.
        
        Args:
            input_data: Input data (can be path to file or DataFrame)
            output_path (str): Path to save results (optional)
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        # Handle various input types
        if isinstance(input_data, str):
            # Check file extension
            if input_data.endswith('.fa') or input_data.endswith('.fasta'):
                predictions = self.predict_from_fasta(input_data)
            elif input_data.endswith('.csv'):
                # Assume CSV has 'sequence' column
                data = pd.read_csv(input_data)
                sequences = data['sequence'].tolist()
                probas = self.predict_batch(sequences)
                
                # Add predictions to DataFrame
                data['splice_probability'] = probas
                data['predicted_splice'] = (probas >= self.threshold).astype(int)
                predictions = data
            else:
                logger.error(f"Unsupported file format: {input_data}")
                return None
        elif isinstance(input_data, pd.DataFrame):
            # Assume DataFrame has 'sequence' column
            sequences = input_data['sequence'].tolist()
            probas = self.predict_batch(sequences)
            
            # Add predictions to DataFrame
            input_data['splice_probability'] = probas
            input_data['predicted_splice'] = (probas >= self.threshold).astype(int)
            predictions = input_data
        else:
            logger.error(f"Unsupported input type: {type(input_data)}")
            return None
            
        # Save results if output path provided
        if output_path is not None:
            self.export_for_surveyor(predictions, output_path)
            
        return predictions
    
    @staticmethod
    def create_surveyor_pipeline(model_path, output_dir, name="foundation_model_pipeline"):
        """
        Create a MetaSpliceAI pipeline configuration for the model.
        
        Args:
            model_path (str): Path to saved model
            output_dir (str): Directory to save pipeline configuration
            name (str): Name for the pipeline
            
        Returns:
            str: Path to pipeline configuration
        """
        pipeline_config = {
            "name": name,
            "description": "Splice site prediction using foundation model",
            "version": "1.0",
            "type": "foundation_model",
            "model_path": model_path,
            "inputs": [
                {
                    "name": "genomic_sequence",
                    "type": "fasta",
                    "description": "Genomic sequence to analyze"
                },
                {
                    "name": "annotations",
                    "type": "gtf",
                    "description": "Gene annotations (optional)",
                    "required": False
                }
            ],
            "outputs": [
                {
                    "name": "splice_predictions",
                    "type": "csv",
                    "description": "Predicted splice sites"
                }
            ],
            "parameters": [
                {
                    "name": "threshold",
                    "type": "float",
                    "default": 0.5,
                    "description": "Classification threshold"
                },
                {
                    "name": "window_size",
                    "type": "int",
                    "default": 10000,
                    "description": "Size of sliding window"
                },
                {
                    "name": "stride",
                    "type": "int",
                    "default": 1000,
                    "description": "Stride for sliding window"
                }
            ]
        }
        
        # Save configuration
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, f"{name}.json")
        
        with open(config_path, 'w') as f:
            json.dump(pipeline_config, f, indent=2)
            
        logger.info(f"Created pipeline configuration at {config_path}")
        return config_path
