import tensorflow as tf
import sys
import os

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

print("--- TensorFlow and System Info ---")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Python Version: {sys.version}")
print("-" * 30)

print("\n--- Checking for GPU ---")
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  - GPU {i}: {gpu}")
        print("\nSUCCESS: TensorFlow can see the GPU(s).")
        print("Attempting to initialize...")
        # Try a minimal computation to force initialization
        with tf.device(gpus[0].name):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
        print("SUCCESS: GPU initialization and a test computation worked.")
    else:
        print("\nWARNING: No GPU devices were found by TensorFlow.")
        print("This could be due to missing drivers, or an incompatible CUDA/cuDNN version.")

except Exception as e:
    print("\n--- ERROR ---")
    print("An error occurred while trying to access or initialize the GPU.")
    print("This strongly suggests a problem with your CUDA or cuDNN installation.")
    print("\nLook for messages above this output (from TensorFlow's C++ backend) for specific library errors (e.g., 'libcudnn.so: cannot open shared object file').")
    print("\nPython-level Error Details:")
    print(e)

print("\n--- End of Report ---")
