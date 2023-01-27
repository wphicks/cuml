from cuml.internals.array import CumlArray
from cuml.internals.logger import warn as log_warn
from cuml.internals.safe_imports import (
    cpu_only_import,
    gpu_only_import,
    safe_import
)
cp = gpu_only_import('cupy')
np = cpu_only_import('numpy')
host_xpy = safe_import('numpy', alt=cp)

def _num_samples(x):
    """Return number of samples in array-like x."""
    x = CumlArray.from_input(x, order='K')
    result = x.shape[0]
    if result == 0:
        raise TypeError(
            f'Singleton array {x} cannot be considered a valid'
            ' collection.'
        )
    return result

def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.
    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    arrays = [
        CumlArray.from_input(X) for X in arrays if X is not None
    ]

    lengths = [_num_samples(X) for X in arrays]
    uniques = host_xpy.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(l) for l in lengths]
        )

def column_or_1d(y, *, dtype=None, warn=False):
    """Ravel column or 1d numpy array, else raises an error.
    Parameters
    ----------
    y : array-like
       Input data.
    dtype : data-type, default=None
        Data type for `y`.
    warn : bool, default=False
       To control display of warnings.
    Returns
    -------
    y : array
       Output data.
    Raises
    ------
    ValueError
        If `y` is not a 1D array or a 2D array with a single row or column.
    """
    y = CumlArray.from_input(
        y,
        convert_to_dtype=dtype
    )

    shape = y.shape
    if len(shape) == 1:
        return y
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            log_warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples, ), for example using ravel()."
            )
        return y.reshape(-1, order='C')
    raise ValueError(
        "y should be a 1d array, got an array of shape {} instead.".format(shape)
    )

def assert_all_finite(X, *, allow_nan=False, **kwargs):
    X = CumlArray.from_input(X)
    xpy = X.mem_type.xpy
    X = X.to_output('array')
    if not xpy.isfinite(xpy.sum(X)):
        has_inf = xpy.isinf(X).any()
        has_nan_error = False if allow_nan else xpy.isnan(X).any()
        if has_inf or has_nan_error:
            if has_nan_error:
                type_err = 'NaN'
            else:
                type_err = (
                    f'infinity or a value too large for'
                    f' {X.dtype}'
                )
            raise ValueError(f'Input contains {type_err})
