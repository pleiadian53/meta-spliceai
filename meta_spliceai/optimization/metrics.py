import numpy as np
import tensorflow as tf
from keras import backend as K



class MulticlassAUC(tf.keras.metrics.AUC):
    """AUC for a single class in a muliticlass problem.

    Parameters
    ----------
    pos_label : int
        Label of the positive class (the one whose AUC is being computed).

    from_logits : bool, optional (default: False)
        If True, assume predictions are not standardized to be between 0 and 1.
        In this case, predictions will be squeezed into probabilities using the
        softmax function.

    sparse : bool, optional (default: True)
        If True, ground truth labels should be encoded as integer indices in the
        range [0, n_classes-1]. Otherwise, ground truth labels should be one-hot
        encoded indicator vectors (with a 1 in the true label position and 0
        elsewhere).

    **kwargs : keyword arguments
        Keyword arguments for tf.keras.metrics.AUC.__init__(). For example, the
        curve type (curve='ROC' or curve='PR').
    """

    def __init__(self, pos_label, from_logits=False, sparse=True, **kwargs):
        super().__init__(**kwargs)

        self.pos_label = pos_label
        self.from_logits = from_logits
        self.sparse = sparse

    def update_state(self, y_true, y_pred, **kwargs):
        """Accumulates confusion matrix statistics.

        Parameters
        ----------
        y_true : tf.Tensor
            The ground truth values. Either an integer tensor of shape
               (n_examples,) (if sparse=True), or 
            a one-hot tensor of shape
               (n_examples, n_classes) (if sparse=False).

        y_pred : tf.Tensor
            The predicted values, a tensor of shape (n_examples, n_classes).

        **kwargs : keyword arguments
            Extra keyword arguments for tf.keras.metrics.AUC.update_state
            (e.g., sample_weight).
        """
        if self.sparse:
            y_true = tf.math.equal(y_true, self.pos_label)
            y_true = tf.squeeze(y_true)
        else:
            y_true = y_true[..., self.pos_label] # index into the column of the "positive" class
            # [[0, 0, 1], 
            #   [0, 1, 0], 
            #   [1, 0 , 1], 
            #   [0, 1, 0]]

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_pred = y_pred[..., self.pos_label]

        super().update_state(y_true, y_pred, **kwargs)


def balanced_accuracy(y_true, y_pred):
	r = recall(y_true, y_pred)
	s = specificity(y_true, y_pred)
	return (r + s)/2.0

def recall(from_logits=True): 
    def recall(y_true, y_pred):
        if from_logits:
            y_pred = K.sigmoid(y_pred)  # Apply sigmoid activation function if from_logits is True
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall_keras = true_positives / (possible_positives + K.epsilon())
        return recall_keras
    return recall

def sensitivity(from_logits=True): 
    return recall(from_logits=from_logits)

def precision(from_logits=True):
    def precision(y_true, y_pred):
        if from_logits:
            y_pred = K.sigmoid(y_pred)  # Apply sigmoid activation function if from_logits is True
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision_keras = true_positives / (predicted_positives + K.epsilon())
        return precision_keras
    return precision

def specificity(from_logits=True): 
    def specificity(y_true, y_pred):
        if from_logits:
            y_pred = K.sigmoid(y_pred)  # Apply sigmoid activation function if from_logits is True
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
        return tn / (tn + fp + K.epsilon())
    return specificity

def negative_predictive_value(from_logits=True): 
    def negative_predictive_value(y_true, y_pred):
        if from_logits:
            y_pred = K.sigmoid(y_pred)  # Apply sigmoid activation function if from_logits is True
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
        return tn / (tn + fn + K.epsilon())
    return negative_predictive_value

def f1(from_logits=True): 
    def f1(y_true, y_pred):
        p = precision(from_logits=from_logits)(y_true, y_pred)
        r = recall(from_logits=from_logits)(y_true, y_pred)
        return 2 * ((p * r) / (p + r + K.epsilon()))
    return f1

def fbeta(y_true, y_pred, beta=2):
    y_pred = K.clip(y_pred, 0, 1)

    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    num = (1 + beta ** 2) * (p * r)
    den = (beta ** 2 * p + r + K.epsilon())
    return K.mean(num / den)


def matthews_correlation_coefficient(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())


def equal_error_rate(y_true, y_pred):
    n_imp = tf.count_nonzero(tf.equal(y_true, 0), dtype=tf.float32) + tf.constant(K.epsilon())
    n_gen = tf.count_nonzero(tf.equal(y_true, 1), dtype=tf.float32) + tf.constant(K.epsilon())

    scores_imp = tf.boolean_mask(y_pred, tf.equal(y_true, 0))
    scores_gen = tf.boolean_mask(y_pred, tf.equal(y_true, 1))

    loop_vars = (tf.constant(0.0), tf.constant(1.0), tf.constant(0.0))
    cond = lambda t, fpr, fnr: tf.greater_equal(fpr, fnr)
    body = lambda t, fpr, fnr: (
        t + 0.001,
        tf.divide(tf.count_nonzero(tf.greater_equal(scores_imp, t), dtype=tf.float32), n_imp),
        tf.divide(tf.count_nonzero(tf.less(scores_gen, t), dtype=tf.float32), n_gen)
    )
    t, fpr, fnr = tf.while_loop(cond, body, loop_vars, back_prop=False)
    eer = (fpr + fnr) / 2

    return eer

def demo_sparse_vs_nonsparse_evaluation(): 
    import tensorflow as tf
    from sklearn.datasets import make_classification

    data = make_classification(n_samples=1000, n_features=20, n_classes=3, n_clusters_per_class=1)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(20)),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(), #or categorical_crossentropy
                optimizer='adam',
                metrics = [tf.keras.metrics.Recall(class_id=1)]
                )

    n = 10 
    print(f"> data prior to categorical encoding:\n{data[1][:n]}\n")
    # NOTE: [1 1 1 1 2 0 1 2 2 0]

    y = tf.keras.utils.to_categorical(data[1], num_classes=3)
    print(f"> after to_categorical():\n{y[:n]}\n")
    # NOTE: [[0. 1. 0.], [0. 1. 0.], [0. 1. 0.], ... [1. 0. 0.]]

    dataset = tf.data.Dataset.from_tensor_slices((data[0], y))
    dataset = dataset.batch(10)

    model.fit(dataset, epochs=10)

    return

def demo_track_auc_for_one_class(): 
    import tensorflow as tf

    sparse = True
    from_logits = False
    unique_category_count = 10

    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    print(f"> shape(X): {x_train.shape}")
    if not sparse: 
        y_train = tf.one_hot(y_train, unique_category_count)
    print(f"> shape(y): {y_train.shape}")

    model = tf.keras.Sequential(layers=[
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=10) if from_logits else tf.keras.layers.Dense(units=10, activation='softmax') 
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    if not sparse: 
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)

    pos_label = 1
    model.compile(
        loss=loss_fn, 
        
        #'sparse_categorical_crossentropy' if sparse else 'categorical_crossentropy', 
        # NOTE: Using the above names doesn't seem to work, why?

        metrics=[MulticlassAUC(pos_label=pos_label, from_logits=from_logits, sparse=sparse, name=f'target_auc'), 
                MulticlassAUC(pos_label=0, from_logits=from_logits, sparse=sparse, name=f'ctrl_auc')],  # Track AUC for class 0 only
    )

    model.fit(x_train, y_train, epochs=10)

    return

def test(): 

    # demo_track_auc_for_one_class()

    # tf.keras.metrics.{Precision, Recall}: Do they work with with sparse mode? i.e. classes are NOT one-hot encoded? 
    demo_sparse_vs_nonsparse_evaluation()

    return

if __name__ == "__main__": 
    test()
