"""
Interpretability tools for splice site prediction models.

This module provides tools for interpreting and visualizing
what features are important for splice site prediction in the models.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import logging
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

class AttentionVisualizer:
    """Visualize attention weights from transformer models."""
    
    def __init__(self, model):
        """
        Initialize attention visualizer.
        
        Args:
            model: Trained model with attention layers
        """
        self.model = model
        
    def _get_attention_model(self):
        """
        Create a model that outputs attention weights.
        
        Returns:
            Model that outputs attention weights
        """
        # For models with attention layers, we need to modify to extract attention weights
        # This implementation depends on model architecture and TF version
        # This is a simplified example that works with the DNATransformer model
        if hasattr(self.model, 'transformer_blocks'):
            inputs = keras.Input(shape=self.model.input_shape[1:])
            attentions = []
            
            # Get embeddings
            x = self.model.embedding(inputs)
            x = self.model.pos_encoding(x)
            
            # Extract attention weights from each transformer block
            for block in self.model.transformer_blocks:
                # Create a temporary model to get attention outputs
                temp_input = keras.Input(shape=(None, self.model.embed_dim))
                attention_output = block.att(temp_input, temp_input, return_attention_scores=True)
                temp_model = keras.Model(inputs=temp_input, outputs=attention_output[1])
                
                # Apply the temp model to get attention scores
                att_weights = temp_model(x)
                attentions.append(att_weights)
                
                # Update x for the next layer using the full block
                x = block(x)
            
            attention_model = keras.Model(inputs=inputs, outputs=attentions)
            return attention_model
        else:
            logger.warning("Model doesn't have standard transformer blocks, can't extract attention weights")
            return None
    
    def visualize_attention(self, sequence_input, head_index=0, layer_index=0, save_path=None):
        """
        Visualize attention weights for a sequence.
        
        Args:
            sequence_input: Input sequence (one-hot encoded)
            head_index (int): Attention head to visualize
            layer_index (int): Transformer layer to visualize
            save_path (str): Path to save visualization
            
        Returns:
            Matplotlib figure
        """
        attention_model = self._get_attention_model()
        if attention_model is None:
            logger.error("Cannot visualize attention: no attention model created")
            return None
            
        # Get attention weights
        attention_weights = attention_model.predict(np.expand_dims(sequence_input, axis=0))
        
        # Extract weights for specific layer and head
        layer_weights = attention_weights[layer_index][0]  # First batch
        head_weights = layer_weights[head_index]  # Specific head
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(head_weights, cmap='viridis')
        
        # Set labels and title
        plt.xlabel('Key position')
        plt.ylabel('Query position')
        plt.title(f'Attention weights (Layer {layer_index}, Head {head_index})')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            
        return ax.get_figure()


class GradientVisualizer:
    """Visualize gradients and saliency maps for model predictions."""
    
    def __init__(self, model):
        """
        Initialize gradient visualizer.
        
        Args:
            model: Trained model
        """
        self.model = model
        
    def compute_gradients(self, input_sequence):
        """
        Compute gradients of output with respect to input sequence.
        
        Args:
            input_sequence: Input sequence (one-hot encoded)
            
        Returns:
            Gradients with respect to input
        """
        # Create a GradientTape to watch the input
        input_tensor = tf.convert_to_tensor(np.expand_dims(input_sequence, axis=0))
        
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            predictions = self.model(input_tensor)
            
        # Get gradients
        gradients = tape.gradient(predictions, input_tensor)
        return gradients[0].numpy()  # Return for first (only) example
    
    def compute_integrated_gradients(self, input_sequence, baseline=None, steps=50):
        """
        Compute integrated gradients for better attribution.
        
        Args:
            input_sequence: Input sequence (one-hot encoded)
            baseline: Baseline input (zeros by default)
            steps (int): Number of steps for approximation
            
        Returns:
            Integrated gradients
        """
        # Create baseline of zeros if not provided
        if baseline is None:
            baseline = np.zeros_like(input_sequence)
            
        # Linearly interpolate between baseline and input
        alphas = np.linspace(0, 1, steps)
        interpolated_inputs = [baseline + alpha * (input_sequence - baseline) for alpha in alphas]
        interpolated_inputs = np.array(interpolated_inputs)
        
        # Compute gradients at each step
        gradients = []
        for interp_input in interpolated_inputs:
            grad = self.compute_gradients(interp_input)
            gradients.append(grad)
            
        # Average gradients and multiply by (input - baseline)
        avg_gradients = np.mean(gradients, axis=0)
        integrated_grads = avg_gradients * (input_sequence - baseline)
        
        return integrated_grads
    
    def visualize_nucleotide_importance(self, sequence, importance_scores, window_size=20, 
                                        smooth=True, window_length=51, polyorder=3, 
                                        save_path=None):
        """
        Visualize importance of each nucleotide in the sequence.
        
        Args:
            sequence (str): Original DNA sequence (ACGT)
            importance_scores (np.ndarray): Importance scores for each position
            window_size (int): Window size for sliding window visualization
            smooth (bool): Whether to apply smoothing
            window_length (int): Window length for Savitzky-Golay filter
            polyorder (int): Polynomial order for Savitzky-Golay filter
            save_path (str): Path to save visualization
            
        Returns:
            Matplotlib figure
        """
        # Calculate importance per position by summing importance across nucleotides
        if importance_scores.ndim > 2:
            # Sum across the nucleotide dimension (usually last dimension)
            position_importance = np.sum(np.abs(importance_scores), axis=-1)
        else:
            position_importance = np.sum(np.abs(importance_scores), axis=1)
            
        # Apply smoothing if requested
        if smooth and len(position_importance) > window_length:
            try:
                position_importance = savgol_filter(position_importance, window_length, polyorder)
            except Exception as e:
                logger.warning(f"Smoothing failed: {e}. Using raw values.")
        
        # Create figure
        plt.figure(figsize=(15, 5))
        
        # Plot importance scores
        plt.plot(position_importance, color='blue')
        
        # Highlight putative splice sites (peaks)
        threshold = np.percentile(position_importance, 95)
        peaks = np.where(position_importance > threshold)[0]
        plt.plot(peaks, position_importance[peaks], 'ro')
        
        # Add sliding windows for context (highlight areas of high importance)
        window_importances = []
        for i in range(len(position_importance) - window_size + 1):
            window_importances.append(np.mean(position_importance[i:i+window_size]))
        
        window_importances = np.array(window_importances)
        window_indices = np.arange(window_size//2, len(position_importance) - window_size//2)
        plt.plot(window_indices, window_importances, color='green', linewidth=2)
        
        # Set labels and title
        plt.xlabel('Sequence Position')
        plt.ylabel('Feature Importance')
        plt.title('Nucleotide Importance for Splice Site Prediction')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            
        return plt.gcf()
    
    def visualize_sequence_logo(self, sequences, importance_scores, window_size=10, 
                               top_k_indices=None, n_logos=5, save_path=None):
        """
        Visualize sequence logos for regions of high importance.
        
        Args:
            sequences (list): List of DNA sequences
            importance_scores (list): List of importance scores for each sequence
            window_size (int): Size of window around important positions
            top_k_indices (list): Indices of top important positions (if None, computed from scores)
            n_logos (int): Number of sequence logos to create
            save_path (str): Path to save visualization
            
        Returns:
            Matplotlib figure
        """
        try:
            import logomaker
        except ImportError:
            logger.error("logomaker package is required for sequence logo visualization")
            print("Please install logomaker: pip install logomaker")
            return None
        
        # Flatten and find top important positions if not provided
        if top_k_indices is None:
            flat_scores = np.array([score.flatten() for score in importance_scores])
            mean_scores = np.mean(np.abs(flat_scores), axis=0)
            top_k_indices = np.argsort(mean_scores)[-n_logos:]
        
        # Create subplot for each important region
        fig, axes = plt.subplots(n_logos, 1, figsize=(10, 3*n_logos))
        if n_logos == 1:
            axes = [axes]
        
        for i, idx in enumerate(top_k_indices):
            # Extract window around important position
            start = max(0, idx - window_size//2)
            end = min(len(sequences[0]), idx + window_size//2)
            
            # Count nucleotides at each position
            counts_matrix = np.zeros((end-start, 4))  # A, C, G, T
            nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
            
            for seq in sequences:
                for j, pos in enumerate(range(start, end)):
                    if pos < len(seq):
                        nuc = seq[pos]
                        if nuc in nuc_map:
                            counts_matrix[j, nuc_map[nuc]] += 1
            
            # Convert to probability
            prob_matrix = counts_matrix / counts_matrix.sum(axis=1, keepdims=True)
            
            # Create DataFrame for logomaker
            df = pd.DataFrame(prob_matrix, columns=['A', 'C', 'G', 'T'])
            
            # Create logo
            logo = logomaker.Logo(df, ax=axes[i])
            logo.style_spines(visible=False)
            axes[i].set_title(f'Sequence motif around position {idx}')
            axes[i].set_xlim([-0.5, end-start-0.5])
            axes[i].set_xticks(range(0, end-start))
            axes[i].set_xticklabels(range(start, end))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            
        return fig


class ShapCalculator:
    """Calculate SHAP values for model predictions."""
    
    def __init__(self, model, background_data=None):
        """
        Initialize SHAP calculator.
        
        Args:
            model: Trained model
            background_data: Background data for SHAP calculation
        """
        self.model = model
        self.background_data = background_data
        
    def calculate_shap_values(self, input_data, n_samples=100):
        """
        Calculate SHAP values for input data.
        
        Args:
            input_data: Input data to explain
            n_samples (int): Number of samples for SHAP approximation
            
        Returns:
            SHAP values
        """
        try:
            import shap
        except ImportError:
            logger.error("shap package is required for SHAP value calculation")
            print("Please install shap: pip install shap")
            return None
        
        # Create explainer
        if self.background_data is not None:
            explainer = shap.DeepExplainer(self.model, self.background_data)
        else:
            # Use a subset of input data as background if not provided
            if len(input_data) > 100:
                background = input_data[:100]
            else:
                background = input_data
            explainer = shap.DeepExplainer(self.model, background)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(input_data, nsamples=n_samples)
        return shap_values
    
    def visualize_shap_summary(self, shap_values, feature_names=None, save_path=None):
        """
        Visualize SHAP summary plot.
        
        Args:
            shap_values: SHAP values
            feature_names (list): Names of features
            save_path (str): Path to save visualization
            
        Returns:
            Matplotlib figure
        """
        try:
            import shap
        except ImportError:
            logger.error("shap package is required for SHAP visualization")
            print("Please install shap: pip install shap")
            return None
        
        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            
        return plt.gcf()


def motif_finder(sequences, labels, kmer_length=6, top_n=10):
    """
    Find enriched sequence motifs in positive examples.
    
    Args:
        sequences (list): DNA sequences
        labels (list): Binary labels (1 for positive, 0 for negative)
        kmer_length (int): Length of k-mers to analyze
        top_n (int): Number of top motifs to return
        
    Returns:
        dict: Dictionary of top motifs and their enrichment scores
    """
    from collections import Counter
    
    # Extract positive and negative sequences
    pos_sequences = [seq for seq, label in zip(sequences, labels) if label == 1]
    neg_sequences = [seq for seq, label in zip(sequences, labels) if label == 0]
    
    # Count k-mers in positive and negative sets
    pos_kmers = Counter()
    for seq in pos_sequences:
        for i in range(len(seq) - kmer_length + 1):
            kmer = seq[i:i+kmer_length]
            if 'N' not in kmer:  # Skip k-mers with unknown nucleotides
                pos_kmers[kmer] += 1
                
    neg_kmers = Counter()
    for seq in neg_sequences:
        for i in range(len(seq) - kmer_length + 1):
            kmer = seq[i:i+kmer_length]
            if 'N' not in kmer:
                neg_kmers[kmer] += 1
    
    # Calculate enrichment scores
    enrichment_scores = {}
    for kmer, pos_count in pos_kmers.items():
        neg_count = neg_kmers.get(kmer, 0)
        
        # Add pseudocounts to avoid division by zero
        pos_freq = (pos_count + 1) / (len(pos_sequences) + 1)
        neg_freq = (neg_count + 1) / (len(neg_sequences) + 1)
        
        # Calculate log2 fold change
        enrichment_scores[kmer] = np.log2(pos_freq / neg_freq)
    
    # Sort by enrichment score and return top motifs
    top_motifs = sorted(enrichment_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return dict(top_motifs)


def visualize_motifs(motif_dict, save_path=None):
    """
    Visualize enriched motifs.
    
    Args:
        motif_dict (dict): Dictionary of motifs and their scores
        save_path (str): Path to save visualization
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(10, 6))
    
    motifs = list(motif_dict.keys())
    scores = list(motif_dict.values())
    
    # Plot motifs and scores
    sns.barplot(x=scores, y=motifs, palette='viridis')
    
    plt.xlabel('Enrichment Score (log2)')
    plt.ylabel('Sequence Motif')
    plt.title('Top Enriched Sequence Motifs')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        
    return plt.gcf()
