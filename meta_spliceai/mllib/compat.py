# compat.py
from sklearn.utils.validation import check_array, _num_samples

def _check_fit_params_v0(X, fit_params, indices=None):
    """Validate and convert `fit_params`.

    Parameters
    ----------
    X : array-like
        The input samples.
    fit_params : dict
        Dictionary containing the parameters passed to the fit method.
    indices : array-like, optional
        Indices of the samples to be used if subsampling.

    Returns
    -------
    fit_params_validated : dict
        Dictionary containing validated and converted `fit_params`.
    """
    fit_params_validated = {}
    for key, value in fit_params.items():
        if value is None:
            continue
        if not hasattr(value, '__len__') and not hasattr(value, 'shape'):
            continue
        if hasattr(value, 'shape'):
            if value.shape[0] != _num_samples(X):
                raise ValueError(f"fit parameter {key} has length {value.shape[0]}, "
                                 f"but X has {X.shape[0]} samples.")
        else:
            if len(value) != _num_samples(X):
                raise ValueError(f"fit parameter {key} has length {len(value)}, "
                                 f"but X has {X.shape[0]} samples.")
        fit_params_validated[key] = check_array(value, accept_sparse=True, force_all_finite=False)
    return fit_params_validated


def _is_arraylike(x):
    """Check if x is array-like."""
    return hasattr(x, "__len__") or hasattr(x, "shape")


def _num_samples(x):
    """Return the number of samples in array-like x."""
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an estimator
        raise TypeError("Expected sequence or array-like, got estimator")
    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        raise TypeError("Expected sequence or array-like, got %s" % type(x))
    if hasattr(x, "shape"):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)


def _make_indexable(iterable):
    """Ensure iterable supports indexing."""
    if hasattr(iterable, "iloc"):
        return iterable
    elif hasattr(iterable, "shape"):
        return iterable
    else:
        return np.array(iterable)

def _safe_indexing(X, indices):
    """Return items or rows from X using indices."""
    if indices is None:
        return X
    if hasattr(X, "iloc"):
        # pandas DataFrame
        return X.iloc[indices]
    elif hasattr(X, "shape"):
        # numpy array or matrix
        return X[indices]
    else:
        # list or other iterable
        return [X[idx] for idx in indices]

def _check_method_params(X, params, indices=None):
    """Check and validate the parameters passed to a specific
    method like `fit`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data array.

    params : dict
        Dictionary containing the parameters passed to the method.

    indices : array-like of shape (n_samples,), default=None
        Indices to be selected if the parameter has the same size as `X`.

    Returns
    -------
    method_params_validated : dict
        Validated parameters. We ensure that the values support indexing.
    """
    method_params_validated = {}
    for param_key, param_value in params.items():
        if not _is_arraylike(param_value) or _num_samples(param_value) != _num_samples(X):
            # Non-indexable pass-through (for now for backward-compatibility).
            # https://github.com/scikit-learn/scikit-learn/issues/15805
            method_params_validated[param_key] = param_value
        else:
            # Any other method_params should support indexing
            # (e.g. for cross-validation).
            method_params_validated[param_key] = _make_indexable(param_value)
            method_params_validated[param_key] = _safe_indexing(
                method_params_validated[param_key], indices
            )

    return method_params_validated


def _check_method_params_v0(X, params, indices=None):
    """Check and validate the parameters passed to a specific
    method like `fit`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data array.

    params : dict
        Dictionary containing the parameters passed to the method.

    indices : array-like of shape (n_samples,), default=None
        Indices to be selected if the parameter has the same size as `X`.

    Returns
    -------
    method_params_validated : dict
        Validated parameters. We ensure that the values support indexing.
    """
    from . import _safe_indexing

    method_params_validated = {}
    for param_key, param_value in params.items():
        if not _is_arraylike(param_value) or _num_samples(param_value) != _num_samples(
            X
        ):
            # Non-indexable pass-through (for now for backward-compatibility).
            # https://github.com/scikit-learn/scikit-learn/issues/15805
            method_params_validated[param_key] = param_value
        else:
            # Any other method_params should support indexing
            # (e.g. for cross-validation).
            method_params_validated[param_key] = _make_indexable(param_value)
            method_params_validated[param_key] = _safe_indexing(
                method_params_validated[param_key], indices
            )

    return method_params_validated
