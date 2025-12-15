import os, sys, time
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from meta_spliceai.utils.utils_doc import (
    print_emphasized, 
    print_with_indent, 
    print_section_separator
)
highlight = print_emphasized
import meta_spliceai.sequence_model.featurize_sequence as fs
from meta_spliceai.sequence_model.data_model import (
    Sequence,
    SequenceMarkers, 
    SequenceCodeBook
)

    
def dcnn_layer(x, n_filters=32, filter_width=2): 
    from tensorflow.keras import layers

    # convolutional layer parameters
    dilation_rates = [2**i for i in range(8)] 

    for dilation_rate in dilation_rates:
        x = layers.Conv1D(filters=n_filters,
                kernel_size=filter_width, 
                padding='causal',
                dilation_rate=dilation_rate)(x)
        # x = layers.BatchNormalization()(x)
    return x

def dcnn_layer_skips(input_seq, n_output=1): 
    """_summary_

    Args:
        input_seq (_type_): A sequence embedding (a tensor) given by layers.Embedding()
        n_output (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_

    Memo: 

    """
    import tensorflow as tf
    # from tensorflow import feature_column
    # from tensorflow import keras
    from tensorflow.keras import layers

    dilation_rates = [1, 2, 4]
    n_filters = 4
    filter_width = 2

    x = input_seq
    skips = []
    for dilation in dilation_rates:

        x = layers.Conv1D(filters = 16,
                                    kernel_size = 1,
                                    padding = 'same',
                                    activation = 'relu')(x)
        
        x_d = layers.Conv1D(filters = n_filters,
                                    kernel_size = filter_width,
                                    padding = 'causal',
                                    dilation_rate = dilation,
                                    activation = tf.keras.activations.swish)(x)
        
        z = layers.Conv1D(filters = 16,
                                    kernel_size = 1,
                                    padding = 'same',
                                    activation = 'relu')(x_d)
        
        x = layers.Add()([x, z])
        
        skips.append(z)
    
    # assemble the skips
    out = layers.Add()(skips)
    output = layers.Conv1D(filters = 1,
                                kernel_size = 1,
                                padding = 'same')(out)
    # out = layers.Flatten()(out)
    # output = layers.Dense(n_output, activation = 'relu')(out)

    return output

def wavenet_layer(x, n_filters=32, filter_width=2, dilation_log_max=5):
    from keras.models import Model
    from keras.layers import Input, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate
    from keras.optimizers import Adam

    # convolutional operation parameters
    # n_filters = 32 
    # filter_width = 2
    dilation_rates = [2**i for i in range(dilation_log_max)] * 2 

    # define an input history series and pass it through a stack of dilated causal convolution blocks. 
    # history_seq = Input(shape=(None, n_dim))
    # x = history_seq

    skips = []
    for dilation_rate in dilation_rates:
        
        # preprocessing - equivalent to time-distributed dense
        x = Conv1D(16, 1, padding='same', activation='relu')(x) 
        
        # filter convolution
        x_f = Conv1D(filters=n_filters,
                    kernel_size=filter_width, 
                    padding='causal',
                    dilation_rate=dilation_rate)(x)
        
        # gating convolution
        x_g = Conv1D(filters=n_filters,
                    kernel_size=filter_width, 
                    padding='causal',
                    dilation_rate=dilation_rate)(x)
        
        # multiply filter and gating branches
        z = Multiply()([Activation('tanh')(x_f),
                        Activation('sigmoid')(x_g)])
        
        # postprocessing - equivalent to time-distributed dense
        z = Conv1D(16, 1, padding='same', activation='relu')(z)
        
        # residual connection
        x = Add()([x, z])    
        
        # collect skip connections
        skips.append(z)

    # add all skip connection outputs 
    out = Activation('elu')(Add()(skips))

    # final time-distributed dense layers 
    out = Conv1D(32, 1, padding='same')(out)
    out = Activation('elu')(out)
    out = Dropout(.2)(out)
    out = Conv1D(16, 1, padding='same')(out)

    return out

def create_two_input_model(optimizer='adam', loss_fn=None, metrics=None,  
            voc_sizes=None, n_dims=None, **kargs):  
    """_summary_

    Args:
        optimizer (str, optional): _description_. Defaults to 'adam'.
        loss_fn (_type_, optional): _description_. Defaults to None.
        metrics (_type_, optional): _description_. Defaults to None.
        voc_sizes (_type_, optional): _description_. Defaults to None.
        n_dims (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_

    Memo: 
        Conv1D, padding
            "causal" results in causal (dilated) convolutions, 
            e.g. output[t] does not depend on input[t+1:]. Useful when modeling temporal data 
            where the model should not violate the temporal order.
    """
    import meta_spliceai.optimization.metrics as pm
    # import tensorflow as tf
    # from tensorflow import keras
    # tf.random.set_seed(42)
    # from tensorflow import feature_column
    from tensorflow.keras import layers
    from keras.regularizers import l2 #l1, l1_l2

    # Parameters: Two embedding layers
    if voc_sizes is None: 
        seq_codes = SequenceCodeBook.get_codebook()
        marker_codes = SequenceMarkers.get_codebook()
        voc_sizes = [len(seq_codes), len(marker_codes)]
    if n_dims is None: n_dims = [64, 64]

    n_tokens_seq = voc_sizes[0]
    n_tokens_marker = voc_sizes[1]
    n_dim_seq = n_dims[0]
    n_dim_marker = n_dims[1]

    # -----------------------------------------------------
    verbose = kargs.get('verbose', 1)
    plot_model = kargs.get('plot_model', True)
    predict_concept = kargs.get('predict_concept', 'nmd')
    n_classes = kargs.get('n_classes', 2)

    # Parameters: Architecture descriptors
    neural_arch_type = kargs.get('neural_arch_type', 'lstm')
    use_bidirectional_arch = True if neural_arch_type.startswith("bi") else False
    merge_type = kargs.get("merge_type", "concat")
    from_logits = kargs.get('from_logits', True)
    sparse = kargs.get("sparse", False)

    # Parameters: Output layer(s)
    output_mode = 'binary'
    single_label = kargs.get('single_label', True)
    if n_classes > 2: output_mode = 'multiclass'
    # Other options: multilabel (single label vs multi-label)

    model_name = kargs.get("model_name", f"multi_input_model-{neural_arch_type}-{merge_type}")
    output_dir = kargs.get("output_dir", os.getcwd()) # save byproducts of model creation e.g. architecture plot

    # Evaluation metrics 
    if metrics is None: 
        if single_label: 
            if n_classes == 1: 
                # One-class classification (e.g. for outlier detection)
                raise NotImplementedError("Coming soon :)")
            if n_classes == 2: 
                metrics = [
                        keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
                        pm.f1, # cm.f1, # f1 score
                        keras.metrics.AUC(name='auc', curve='ROC'),
                        # pm.balanced_accuracy, 
                    ]
            else: # n_classes > 2 
                metrics = [
                    keras.metrics.AUC(name='prc', curve='PR', from_logits=from_logits, multi_label=False), # precision-recall curve
                    keras.metrics.AUC(name='auc', from_logits=from_logits, multi_label=False),
                ]
        else: 
            # multi-label
            raise NotImplementedError("Coming soon :)")

    if loss_fn is None: 
        if n_classes == 2: 
            loss_fn = keras.losses.BinaryCrossentropy(from_logits=from_logits)
        else: 
            if sparse: 
                loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
            else: 
                loss_fn = keras.losses.CategoricalCrossentropy(from_logits=from_logits)

        print(f"(create_two_input_model) Arch type: {neural_arch_type}, loss function:\n{loss_fn.name}\n")

    model_spec = {'input1': 'sequence', 'input2': 'marker', 'output1': predict_concept}

    # ----- Define Neural Architectures -----
    highlight(f"(create_two_input_model) Creating neural network of type: {neural_arch_type}, merge type: {merge_type}", symbol='-')

    transcript_input = keras.Input(shape=(None,), name="sequence")  # the name is used to match training data input
    # NOTE: 
    #   - None referes to a variable-length sequence of ints
    #   - batch dimension is not specified; None does not refers to the batch dimension

    marker_input = keras.Input(shape=(None,), name="marker")
    # NOTE: if instead of shape=(None, ), we specify 
    #       keras.Input(shape=(11,), name="marker")
    #       => expected input shape=(None, 11), any batch with shape (x, 11) will be compatible 
    #          e.g. a batch of shape=(32, 11), where 32 is the batch size => compatible

    # Embed each char in the transcript a 64-dimensional vector
    transcript_embedding = layers.Embedding(n_tokens_seq, n_dim_seq)(transcript_input)
    # Embed each char in the text into a 64-dimensional vector
    marker_embedding = layers.Embedding(n_tokens_marker, n_dim_marker)(marker_input) # input_length
    # NOTE: suppose n_tokens_marker = 10, n_dim_marker = 64 
    #       - the model takes as input an integer matrix of size (batch, input_length), where input_length is None, which can be an arbitrary length
    #       - the largest number in the matrix will range from 0 to 9 (vocab size=10)
    #       - each token, represented by an integer, is embedded into a 64-D vector
    #       - suppose that the input_length is 100, then 
    #         model.output_shape is (None, 100, 64), where 100 is the sequence length, and 64 is the token's vector dimension

    if use_bidirectional_arch: 
        t = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(transcript_embedding)
        t = layers.Bidirectional(layers.LSTM(32))(t)

        m = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(marker_embedding)
        m = layers.Bidirectional(layers.LSTM(32))(m)
        # NOTE: shape(m): (None, 32 * 2)

        t_model = keras.Model(inputs=transcript_input, outputs=t)
        m_model = keras.Model(inputs=marker_input, outputs=m)
    else: 

        if neural_arch_type.lower() in ('lstm', ): 
            # Reduce sequence of embedded words in the title into a single 128-dimensional vector
            t = layers.LSTM(64)(transcript_embedding)
            print(f"... embedding -> LSTM: {transcript_embedding.shape} -> {t.shape}")
            # NOTE: (None, None, 32) -> (None, 64)
            #       say input sequence has a length of 10
            #       (None, 10, 32) -> (None, 64)  i.e. (10, 32) is compiled into a 64-D vector representing the entire sequenc of len=10

            # Reduce sequence of embedded words in the body into a single 32-dimensional vector
            m = layers.LSTM(64)(marker_embedding)

            t_model = keras.Model(inputs=transcript_input, outputs=t)
            m_model = keras.Model(inputs=marker_input, outputs=m)
        elif neural_arch_type.lower() in ('cnn1', 'cnn1_base' ):
            
            # create_two_input_model: configure neural architectural parameters
            neural_arch_params = kargs.get('neural_arch_params', {})
            if neural_arch_params: 
                n_filters = [24, 32]
                n_filters = neural_arch_params.get('n_filters', n_filters) 
                assert len(n_filters) >= 1
                k_sizes = [6, 3]
                k_sizes = neural_arch_params.get('kernel_sizes', k_sizes) 
                print(f"[create_model] CNN specs")

            padding = 'same' # 'causal', 'same', 'valid'

            # Define transcript sequence model ---------
            conv1 = layers.Conv1D(filters=n_filters[0], kernel_size=k_sizes[0],
                                    strides=1, padding=padding,
                                    activation="elu") # input_shape=[n_tokens_seq, n_dim_seq]
            # kernel_regularizer = keras.regularizers.l2( l=0.01)
            # NOTE: adding `kernel_regularizer`` usually doesn't help 

            # conv2 = layers.Conv1D(filters=64, kernel_size=3,
            #                         strides=1, padding=padding,
            #                         activation="elu") # kernel_regularizer = keras.regularizers.l2( l=0.01)
            # dropout1 = layers.Dropout(0.3)
            # maxpool1 = layers.MaxPooling1D(2)
            bn1 = layers.BatchNormalization() # epsilon=1e-06, mode=0, momentum=0.9, weights=None
            global_pool = layers.GlobalAveragePooling1D()
            # dropout2 = layers.Dropout(0.3)
            # dense_t = layers.Dense(10, activation="relu", kernel_regularizer=l2(0.01))
            # ----------------------------------
            
            # Build network using the defined layers ----
            t = conv1(transcript_embedding) # 32 units
            # t = conv2(t)
            # t = maxpool1(t)
            # t = bn1(t)
            t = global_pool(t)
            # t = tf.reduce_max(t, axis=1)

            t_model = keras.Model(inputs=transcript_input, outputs=t, name='transcript_sequence')

            # Define marker sequence model ------ 
            conv1_m = layers.Conv1D(filters=n_filters[0], kernel_size=k_sizes[0],
                                    strides=1, padding=padding,
                                    activation="elu", kernel_regularizer = keras.regularizers.l2( l=0.01)) # input_shape=[n_tokens_seq, n_dim_seq]
            conv2_m = layers.Conv1D(filters=64, kernel_size=3,
                                    strides=1, padding=padding,
                                    activation="elu", kernel_regularizer = keras.regularizers.l2( l=0.01))
            # dropout1_m = layers.Dropout(0.3)
            # maxpool1_m = layers.MaxPooling1D(2)
            bn1_m = layers.BatchNormalization() # epsilon=1e-06, mode=0, momentum=0.9, weights=None
            global_pool_m = layers.GlobalAveragePooling1D()
            # dropout2_m = layers.Dropout(0.3)
            # dense_m = layers.Dense(10, activation="relu", kernel_regularizer=l2(0.01))
            # ----------------------------------
            
            m = conv1_m(marker_embedding)
            # m = conv2_m(m)
            # m = bn1_m(m)
            m = global_pool_m(m)
            # m = tf.reduce_max(m, axis=1)

            m_model = keras.Model(inputs=marker_input, outputs=m, name='marker_sequence')

        elif neural_arch_type.lower() in ('cnn2', 'cnn2_base', 'cnn', ):

            # create_two_input_model: configure neural architectural parameters
            neural_arch_params = kargs.get('neural_arch_params', {})
            if neural_arch_params: 
                n_filters = [24, 32]
                n_filters = neural_arch_params.get('n_filters', n_filters) 
                assert len(n_filters) >= 2
                k_sizes = [5, 3]
                k_sizes = neural_arch_params.get('kernel_sizes', k_sizes) 
                print(f"[create_model] CNN specs")

            padding = 'same' # 'causal', 'same', 'valid'
            
            # Define layers/network components for transcript sequences ---- 
            conv1 = layers.Conv1D(filters=n_filters[0], kernel_size=k_sizes[0],
                                    strides=1, padding=padding,
                                    activation="elu") # input_shape=[n_tokens_seq, n_dim_seq]
            maxpool1 = layers.MaxPooling1D(2)
            # conv2 = layers.Conv1D(filters=32, kernel_size=5,
            #                         strides=1, padding=padding,
            #                         activation="relu")
            # maxpool2 = layers.MaxPooling1D(2)
            bn1 = layers.BatchNormalization() # epsilon=1e-06, mode=0, momentum=0.9, weights=None
            dropout1 = layers.Dropout(0.3)

            conv3 = layers.Conv1D(filters=n_filters[1], kernel_size=k_sizes[1],
                                    strides=1, padding=padding,
                                    activation="elu")
            # conv4 = layers.Conv1D(filters=32, kernel_size=3,
            #                         strides=1, padding=padding,
            #                         activation="relu")
            maxpool3 = layers.MaxPooling1D(2)
            bn3 = layers.BatchNormalization()
            dropout3 = layers.Dropout(0.4)

            global_pool = layers.GlobalAveragePooling1D()
            dropout = layers.Dropout(0.3)
            dense_t = layers.Dense(10, activation="relu", kernel_regularizer=l2(0.01))

            # Build network using the defined layers ----
            t = conv1(transcript_embedding) # 
            # t = conv2(t)
            t = maxpool1(t)
            t = bn1(t)
            # t = dropout1(t)

            t = conv3(t) # 
            t = maxpool3(t)
            t = bn3(t)
            t = dropout3(t)

            t = global_pool(t)
            # t = dense_t(t)

            # transcript_features = t
            t_model = keras.Model(inputs=transcript_input, outputs=t)
            # --------------------------------------------------
             
            padding = 'same' # 'causal', 'same', 'valid'
            # Define layers/network components for marker sequences ---- 
            conv1_m = layers.Conv1D(filters=n_filters[0], kernel_size=k_sizes[0],
                                    strides=1, padding=padding,
                                    activation="elu") # input_shape=[n_tokens_seq, n_dim_seq]
            maxpool1_m = layers.MaxPooling1D(2)
            # conv2_m = layers.Conv1D(filters=32, kernel_size=5,
            #                         strides=1, padding=padding,
            #                         activation="relu")
            # maxpool2_m = layers.MaxPooling1D(2)
            bn1_m = layers.BatchNormalization()
            dropout1_m = layers.Dropout(0.3)

            conv3_m = layers.Conv1D(filters=n_filters[1], kernel_size=k_sizes[1],
                                    strides=1, padding=padding,
                                    activation="elu")
            # conv4_m = layers.Conv1D(filters=32, kernel_size=3,
            #                         strides=1, padding=padding,
            #                         activation="relu")
            maxpool3_m = layers.MaxPooling1D(2)
            bn3_m = layers.BatchNormalization()
            dropout3_m = layers.Dropout(0.4)

            global_pool_m = layers.GlobalAveragePooling1D()
            dropout_m = layers.Dropout(0.3)
            dense_m = layers.Dense(10, activation="relu", kernel_regularizer=l2(0.01))

            # Build network using the defined layers ----
            v = conv1_m(marker_embedding)
            # v = conv2_m(v)
            v = maxpool1_m(v)
            v = bn1_m(v)
            # v = dropout1_m(v)

            v = conv3_m(v)
            v = maxpool3_m(v)
            v = bn3_m(v)  
            v = dropout3_m(v)

            v = global_pool_m(v) # [None, None, 32] -> [None, 32]
            # v = dense_m(v)

            # marker_features = v
            m_model = keras.Model(inputs=marker_input, outputs=v)
            # --------------------------------------------

            # v = global_pool_m(v)
            # v = dropout_m(v)
            # marker_features = dense_m(v)
        elif neural_arch_type in ('dcnn', ): 
            t = dcnn_layer(transcript_embedding)
            t = layers.GlobalAveragePooling1D()(t)
            # t = layers.Dropout(0.3)(t)
            t = layers.Dense(10, activation="relu", kernel_regularizer=l2(0.01))(t)

            m = dcnn_layer(marker_embedding)
            m = layers.GlobalAveragePooling1D()(m)
            # m = layers.Dropout(0.3)(m)
            m = layers.Dense(10, activation="relu", kernel_regularizer=l2(0.01))(m)

            # Add additional layers ...

            t_model = keras.Model(inputs=transcript_input, outputs=t) # transcript_input -> transcript_embedding -> DCNN -> t -> ... -> t
            m_model = keras.Model(inputs=marker_input, outputs=m)
            
            # transcript_features = t
            # marker_features = m
        elif neural_arch_type in ( 'wavenet', 'dcnn2', ):
            log_max = kargs.get('dilation_log_max', 4)
            print("> wavenet params")
            print(f"... dilation rates at most: {2**log_max}")
            t = wavenet_layer(transcript_embedding, n_filters=32, filter_width=2, dilation_log_max=log_max) 
            # NOTE: shape(t): (None, seq_length, 1)
            t = layers.GlobalAveragePooling1D()(t)
            t_model = keras.Model(inputs=transcript_input, outputs=t)

            m = wavenet_layer(marker_embedding, n_filters=32, filter_width=2, dilation_log_max=log_max) 
            m = layers.GlobalAveragePooling1D()(m)
            m_model = keras.Model(inputs=marker_input, outputs=m)

    # Merge all available features into a single large vector via concatenation
    if merge_type.startswith('concat'): 
        # merged = layers.concatenate([transcript_features, marker_features])
        merged = layers.concatenate([t_model.output, m_model.output])
        print(f"... concatenating: {t_model.output.shape} || {m_model.output.shape} => {merged.shape}")
        # NOTE: (None, 10) || (None, 10) => (None, 20)

    elif merge_type.startswith('multi'):
        # merged = layers.Multiply()([transcript_features, marker_features])
        merged = layers.Multiply()([t_model.output, m_model.output])
        # print(f"... combining by multiplying => shape(x)={x.shape}") # (None, None, 32) x (None, None, 32) => (None, None, 32)

    # x = layers.Flatten()(x)
    # print(f"... flatten(x) => shape(x): {x.shape}")
    # x = layers.GlobalAveragePooling1D()(merged)
    # NOTE: (None, 64)
    #       suppose that each token is repr by 64-D vector and the sequence length = 10
    #       => global averge pooling, takes the average of these 10 64-D vectors to give only single 64-D vector

    x = merged # feature vector after merging two-input cascading layers

    # Classifier "neck" -------------
    if neural_arch_type.find('base') > 0: # ['cnn1_base', 'cnn2_base', ]: 
        pass 
    else: 
        x = layers.Dense(10, activation='selu', kernel_regularizer=l2(0.01))(x) # kernel_regularizer=l2(0.01)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(.5)(x)

    # Classifier "head" --------------
    if output_mode.startswith('bin'): 
        if from_logits: 
            concept_output = layers.Dense(1, name=predict_concept)(x)
            # NOTE: activation='sigmoid' is not needed because 
            #       tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else: 
            concept_output = layers.Dense(1, name=predict_concept, activation='sigmoid')(x) 
    elif output_mode.startswith('multi'):
        if from_logits: 
            concept_output = layers.Dense(n_classes, name=predict_concept)(x)
        else: 
            concept_output = layers.Dense(n_classes, name=predict_concept, activation='softmax')(x)

    # Instantiate an end-to-end model predicting both priority and department
    model = keras.Model(
        # inputs=[transcript_input, marker_input],
        inputs=[t_model.input, m_model.input], 
        outputs=[concept_output, ], name='T-Classifier'
    )

    # Plot model 
    if plot_model: 
        output_path = os.path.join(output_dir, f"{model_name}-{predict_concept}.png")
        print(f"[build model] Saving model architecture at:\n{output_path}\n")
        keras.utils.plot_model(model, output_path, show_shapes=True)
    
    # Display the model's architecture
    if verbose: 
        model.summary()

    model.compile(
        optimizer=optimizer, # keras.optimizers.RMSprop(1e-3),
        loss={
            predict_concept: loss_fn, # keras.losses.BinaryCrossentropy(from_logits=True),
            # ge_concept: keras.losses.CategoricalCrossentropy(from_logits=True),
        }, 
        metrics=metrics
    )

    return model, model_spec


def infer_embedding_layer_params(model): 
    # Test
    assert model.layers[0].name == 'sequence'
    assert model.layers[1].name == 'marker'

    # input_0 = base_model.layers[0].name
    # input_1 = base_model.layers[1].name
    embed_names = {'sequence': 'embedding', 'marker': 'embedding_1'}
    embeddings = {}
    for layer in model.layers:
        for s in ['sequence', 'marker', ]: 
            if layer.name == embed_names[s]: 
                embeddings[s] = layer.get_weights()[0]

    vocab_size_seq = embeddings['sequence'].shape[0]
    vocab_size_marker = embeddings['marker'].shape[0]
    n_dim_seq = embeddings['sequence'].shape[1]
    n_dim_marker = embeddings['marker'].shape[1]
    return {'voc_sizes': [vocab_size_seq, vocab_size_marker], 'n_dims': [n_dim_seq, n_dim_marker]}  


def create_cnn_branch(input_tensor, n_filters, k_sizes, dropout_rates, padding='same'):
    """ Create a CNN branch for given input_tensor. """
    # import tensorflow as tf
    from tensorflow.keras import layers # regularizers

    x = input_tensor
    for i in range(len(n_filters)):
        # Convolutional layer
        x = layers.Conv1D(filters=n_filters[i], kernel_size=k_sizes[i],
                        strides=1, padding=padding)(x)
        # Max pooling
        x = layers.MaxPooling1D(2)(x) # expect 3D tensor
        # Batch normalization
        x = layers.BatchNormalization()(x)
        # Activation
        x = layers.Activation('elu')(x)
        # Dropout
        x = layers.Dropout(dropout_rates[i])(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    return x 

# Enumeration for output modes
class OutputMode:
    BINARY = "binary"
    MULTI_CLASS = "multi_class"

def classifier_head(input_tensor, output_mode, n_classes=1, from_logits=True, bias_init=None):
    """ Define the classifier head based on the output_mode. """
    # from tensorflow.keras import layers
    if output_mode == OutputMode.BINARY:
        if from_logits:
            return layers.Dense(1, bias_initializer=bias_init)(input_tensor)
        else:
            return layers.Dense(1, activation='sigmoid', bias_initializer=bias_init)(input_tensor)
    elif output_mode == OutputMode.MULTI_CLASS:
        if from_logits:
            return layers.Dense(n_classes)(input_tensor)
        else:
            return layers.Dense(n_classes, activation='softmax')(input_tensor)
    else:
        raise ValueError(f"Unknown output mode: {output_mode}")