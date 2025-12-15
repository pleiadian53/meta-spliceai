import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Cropping1D, Dense
from tensorflow.keras.models import Model
import numpy as np



# Example one-hot encoding function
def one_hot_encode(sequence):
    mapping = {'A': [1, 0, 0, 0],
               'C': [0, 1, 0, 0],
               'G': [0, 0, 1, 0],
               'T': [0, 0, 0, 1],
               'N': [0, 0, 0, 0]}
    return np.array([mapping[base] for base in sequence])

# Input sequence
input_sequence = 'ACGTACGTACGT'
context = 10000
crop_size = 5000

# Padding the input sequence to match the context length
padded_sequence = 'N'*(context//2) + input_sequence + 'N'*(context//2)
x = one_hot_encode(padded_sequence)[None, :]  # Shape: (1, context + len(input_sequence), 4)

# Define the model
input_layer = Input(shape=(x.shape[1], 4))
# NOTE: 
# - The Input(shape=(x.shape[1], 4)) specifies the shape of each individual data instance (excluding the batch size)
# - x.shape[1] corresponds to the length of the input sequence after padding (in this case, the total sequence length)
# - The number 4 refers to the four channels corresponding to the one-hot encoded nucleotides (A, C, G, T)

# Example convolutional layer
conv_layer = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(input_layer)
# A Conv1D layer simulates SpliceAI’s convolutional layers, which capture local patterns in the sequence.

# Cropping layer to remove padding
cropping_layer = Cropping1D(cropping=(crop_size, crop_size))(conv_layer)
# The Cropping1D layer is applied to crop off the edges of the padded sequence. In SpliceAI, this typically removes 5,000 nucleotides from each end, focusing the model on the central part of the sequence.

# Example dense layer to simulate prediction
output_layer = Dense(3, activation='softmax')(cropping_layer)
# A Dense layer is used here to simulate the prediction output. In SpliceAI, this would correspond to the output layer predicting donor, acceptor, or neither probabilities.

# Build the model
model = Model(inputs=input_layer, outputs=output_layer)

# Summarize the model
model.summary()

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_1 (InputLayer)        [(None, 10012, 4)]        0         
#                                                               
#  conv1d (Conv1D)             (None, 10012, 32)         672       
#                                                               
#  cropping1d (Cropping1D)     (None, 12, 32)            0         
#                                                                
#  dense (Dense)               (None, 12, 3)             99        
#                                                                
# =================================================================

# Running the model on the input sequence
predictions = model.predict(x)
print(f"Shape of predictions: {predictions.shape}")  # Shape of predictions: (1, 12, 3)

# NOTE: predictions.shape: 
#    - This will show the shape of the output, which should be shorter than the original input sequence 
#      by the amount cropped (i.e., context - 2 * crop_size + len(input_sequence)).

