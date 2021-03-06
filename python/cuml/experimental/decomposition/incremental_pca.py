#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cupy as cp
import scipy
import numbers

from cuml.common import with_cupy_rmm
from cuml.common import input_to_cuml_array
from cuml.common.array import CumlArray

from cuml.decomposition import PCA


class IncrementalPCA(PCA):
    """
    Based on sklearn.decomposition.IncrementalPCA from scikit-learn 0.23.1

    Incremental principal components analysis (IPCA).
    Linear dimensionality reduction using Singular Value Decomposition of
    the data, keeping only the most significant singular vectors to
    project the data to a lower dimensional space. The input data is
    centered but not scaled for each feature before applying the SVD.
    Depending on the size of the input data, this algorithm can be much
    more memory efficient than a PCA, and allows sparse input.
    This algorithm has constant memory complexity, on the order of
    ``batch_size * n_features``, enabling use of np.memmap files without
    loading the entire file into memory. For sparse matrices, the input
    is converted to dense in batches (in order to be able to subtract the
    mean) which avoids storing the entire dense matrix at any one time.
    The computational overhead of each SVD is
    ``O(batch_size * n_features ** 2)``, but only 2 * batch_size samples
    remain in memory at a time. There will be ``n_samples / batch_size``
    SVD computations to get the principal components, versus 1 large SVD
    of complexity ``O(n_samples * n_features ** 2)`` for PCA.


    Examples
    ---------

    .. code-block:: python
        from cuml.decomposition import IncrementalPCA
        import cupy as cp

        X = cp.sparse.random(1000, 5, format='csr', density=0.07)
        ipca = IncrementalPCA(n_components=2, batch_size=200)
        ipca.fit(X)

        print("Components: \n", ipca.components_)

        print("Singular Values: ", ipca.singular_values_)

        print("Explained Variance: ", ipca.explained_variance_)

        print("Explained Variance Ratio: ",
            ipca.explained_variance_ratio_)

        print("Mean: ", ipca.mean_)

        print("Noise Variance: ", ipca.noise_variance_)

    Output:

    .. code-block:: python
        Components:
        [[ 0.40465797  0.70924681 -0.46980153 -0.32028596 -0.09962083]
        [ 0.3072285  -0.31337166 -0.21010504 -0.25727659  0.83490926]]

        Singular Values: [4.67710479 4.0249979 ]

        Explained Variance: [0.02189721 0.01621682]

        Explained Variance Ratio: [0.2084041  0.15434174]

        Mean: [0.03341744 0.03796517 0.03316038 0.03825872 0.0253353 ]

        Noise Variance: 0.0049539530909571425

    Parameters
    ----------
    handle : cuml.Handle
        If it is None, a new one is created just for this class
    n_components : int or None, (default=None)
        Number of components to keep. If ``n_components `` is ``None``,
        then ``n_components`` is set to ``min(n_samples, n_features)``.
    whiten : bool, optional
        If True, de-correlates the components. This is done by dividing them by
        the corresponding singular values then multiplying by sqrt(n_samples).
        Whitening allows each component to have unit variance and removes
        multi-collinearity. It might be beneficial for downstream
        tasks like LinearRegression where correlated features cause problems.
    copy : bool, (default=True)
        If False, X will be overwritten. ``copy=False`` can be used to
        save memory but is unsafe for general use.
    batch_size : int or None, (default=None)
        The number of samples to use for each batch. Only used when calling
        ``fit``. If ``batch_size`` is ``None``, then ``batch_size``
        is inferred from the data and set to ``5 * n_features``, to provide a
        balance between approximation accuracy and memory consumption.
    verbose : int or boolean (default = False)
        Logging level
    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Components with maximum variance.
    explained_variance_ : array, shape (n_components,)
        Variance explained by each of the selected components.
    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If all components are stored, the sum of explained variances is equal
        to 1.0.
    singular_values_ : array, shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.
    mean_ : array, shape (n_features,)
        Per-feature empirical mean, aggregate over calls to ``partial_fit``.
    var_ : array, shape (n_features,)
        Per-feature empirical variance, aggregate over calls to
        ``partial_fit``.
    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf.
    n_components_ : int
        The estimated number of components. Relevant when
        ``n_components=None``.
    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to fit, but increments across ``partial_fit`` calls.
    batch_size_ : int
        Inferred batch size from ``batch_size``.
    Notes
    -----
    Implements the incremental PCA model from:
    *D. Ross, J. Lim, R. Lin, M. Yang, Incremental Learning for Robust Visual
    Tracking, International Journal of Computer Vision, Volume 77, Issue 1-3,
    pp. 125-141, May 2008.*
    See https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf
    This model is an extension of the Sequential Karhunen-Loeve Transform from:
    *A. Levy and M. Lindenbaum, Sequential Karhunen-Loeve Basis Extraction and
    its Application to Images, IEEE Transactions on Image Processing, Volume 9,
    Number 8, pp. 1371-1374, August 2000.*
    See https://www.cs.technion.ac.il/~mic/doc/skl-ip.pdf
    We have specifically abstained from an optimization used by authors of both
    papers, a QR decomposition used in specific situations to reduce the
    algorithmic complexity of the SVD. The source for this technique is
    *Matrix Computations, Third Edition, G. Holub and C. Van Loan, Chapter 5,
    section 5.4.4, pp 252-253.*. This technique has been omitted because it is
    advantageous only when decomposing a matrix with ``n_samples`` (rows)
    >= 5/3 * ``n_features`` (columns), and hurts the readability of the
    implemented algorithm. This would be a good opportunity for future
    optimization, if it is deemed necessary.
    References
    ----------
    D. Ross, J. Lim, R. Lin, M. Yang. Incremental Learning for Robust Visual
    Tracking, International Journal of Computer Vision, Volume 77,
    Issue 1-3, pp. 125-141, May 2008.
    G. Golub and C. Van Loan. Matrix Computations, Third Edition, Chapter 5,
    Section 5.4.4, pp. 252-253.
    """
    def __init__(self, handle=None, n_components=None, *, whiten=False,
                 copy=True, batch_size=None, verbose=None,
                 output_type=None):

        super(IncrementalPCA, self).__init__(handle=handle,
                                             n_components=n_components,
                                             whiten=whiten, copy=copy,
                                             verbose=verbose,
                                             output_type=output_type)
        self.batch_size = batch_size
        self._hyperparams = ["n_components", "whiten", "copy", "batch_size"]
        self._cupy_attributes = True
        self._sparse_model = True

    @with_cupy_rmm
    def fit(self, X, y=None):
        """Fit the model with X, using minibatches of size batch_size.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        y : Ignored
        Returns
        -------
        self : object
            Returns the instance itself.
        """

        self._set_output_type(X)

        self.n_samples_seen_ = 0
        self._mean_ = .0
        self.var_ = .0

        if scipy.sparse.issparse(X) or cp.sparse.issparse(X):
            X = _validate_sparse_input(X)
        else:
            X, n_samples, n_features, self.dtype = \
                input_to_cuml_array(X, order='K',
                                    check_dtype=[cp.float32, cp.float64])

            # NOTE: While we cast the input to a cupy array here, we still
            # respect the `output_type` parameter in the constructor. This
            # is done by PCA, which IncrementalPCA inherits from. PCA's
            # transform and inverse transform convert the output to the
            # required type.
            X = X.to_output(output_type='cupy')

        n_samples, n_features = X.shape

        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        for batch in _gen_batches(n_samples, self.batch_size_,
                                  min_batch_size=self.n_components or 0):
            X_batch = X[batch]
            if cp.sparse.issparse(X_batch):
                X_batch = X_batch.toarray()

            self.partial_fit(X_batch, check_input=False)

        return self

    @with_cupy_rmm
    def partial_fit(self, X, y=None, check_input=True):
        """Incremental fit with X. All of X is processed as a single batch.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        check_input : bool
            Run check_array on X.
        y : Ignored
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if check_input:
            if scipy.sparse.issparse(X) or cp.sparse.issparse(X):
                raise TypeError(
                    "IncrementalPCA.partial_fit does not support "
                    "sparse input. Either convert data to dense "
                    "or use IncrementalPCA.fit to do so in batches.")

            self._set_output_type(X)

            X, n_samples, n_features, self.dtype = \
                input_to_cuml_array(X, order='K',
                                    check_dtype=[cp.float32, cp.float64])
            X = X.to_output(output_type='cupy')
        else:
            n_samples, n_features = X.shape

        if not hasattr(self, '_components_'):
            self._components_ = None

        if self.n_components is None:
            if self._components_ is None:
                self.n_components_ = min(n_samples, n_features)
            else:
                self.n_components_ = self._components_.shape[0]
        elif not 1 <= self.n_components <= n_features:
            raise ValueError("n_components=%r invalid for n_features=%d, need "
                             "more rows than columns for IncrementalPCA "
                             "processing" % (self.n_components, n_features))
        elif not self.n_components <= n_samples:
            raise ValueError("n_components=%r must be less or equal to "
                             "the batch number of samples "
                             "%d." % (self.n_components, n_samples))
        else:
            self.n_components_ = self.n_components

        if (self._components_ is not None) and (self._components_.shape[0] !=
                                                self.n_components_):
            raise ValueError("Number of input features has changed from %i "
                             "to %i between calls to partial_fit! Try "
                             "setting n_components to a fixed value." %
                             (self._components_.shape[0], self.n_components_))

        if not self._cupy_attributes:
            self._cumlarray_to_cupy_attrs()
            self._cupy_attributes = True

        # This is the first partial_fit
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = 0
            self._mean_ = .0
            self.var_ = .0

        # Update stats - they are 0 if this is the first step
        col_mean, col_var, n_total_samples = \
            _incremental_mean_and_var(
                X, last_mean=self._mean_, last_variance=self.var_,
                last_sample_count=cp.repeat(cp.asarray([self.n_samples_seen_]),
                                            X.shape[1]))
        n_total_samples = n_total_samples[0]

        # Whitening
        if self.n_samples_seen_ == 0:
            # If it is the first step, simply whiten X
            X = X - col_mean
        else:
            col_batch_mean = cp.mean(X, axis=0)
            X = X - col_batch_mean
            # Build matrix of combined previous basis and new data
            mean_correction = \
                cp.sqrt((self.n_samples_seen_ * n_samples) /
                        n_total_samples) * (self._mean_ - col_batch_mean)
            X = cp.vstack((self._singular_values_.reshape((-1, 1)) *
                           self._components_, X, mean_correction))

        U, S, V = cp.linalg.svd(X, full_matrices=False)
        U, V = _svd_flip(U, V, u_based_decision=False)
        explained_variance = S ** 2 / (n_total_samples - 1)
        explained_variance_ratio = S ** 2 / cp.sum(col_var * n_total_samples)

        self.n_samples_seen_ = n_total_samples
        self._components_ = V[:self.n_components_]
        self._singular_values_ = S[:self.n_components_]
        self._mean_ = col_mean
        self.var_ = col_var
        self._explained_variance_ = explained_variance[:self.n_components_]
        self._explained_variance_ratio_ = \
            explained_variance_ratio[:self.n_components_]
        if self.n_components_ < n_features:
            self._noise_variance_ = \
                explained_variance[self.n_components_:].mean()
        else:
            self._noise_variance_ = 0.

        if self._cupy_attributes:
            self._cupy_to_cumlarray_attrs()
            self._cupy_attributes = False

        return self

    @with_cupy_rmm
    def transform(self, X, convert_dtype=False):
        """Apply dimensionality reduction to X.
        X is projected on the first principal components previously extracted
        from a training set, using minibatches of size batch_size if X is
        sparse.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        convert_dtype : bool, optional (default = False)
            When set to True, the transform method will automatically
            convert the input to the data type which was used to train the
            model. This will increase memory used for the method.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """

        if scipy.sparse.issparse(X) or cp.sparse.issparse(X):
            out_type = self._get_output_type(X)

            X = _validate_sparse_input(X)

            n_samples = X.shape[0]
            output = []
            for batch in _gen_batches(n_samples, self.batch_size_,
                                      min_batch_size=self.n_components or 0):
                output.append(super().transform(X[batch]))
            output, _, _, _ = \
                input_to_cuml_array(cp.vstack(output), order='K')

            return output.to_output(out_type)
        else:
            return super().transform(X)

    def get_param_names(self):
        return self._hyperparams

    def _cupy_to_cumlarray_attrs(self):
        self._components_ = CumlArray(self._components_.copy())
        self._mean_ = CumlArray(self._mean_)
        self._noise_variance_ = CumlArray(self._noise_variance_)
        self._singular_values_ = CumlArray(self._singular_values_)
        self._explained_variance_ = CumlArray(self._explained_variance_.copy())
        self._explained_variance_ratio_ = \
            CumlArray(self._explained_variance_ratio_)

    def _cumlarray_to_cupy_attrs(self):
        self._components_ = self._components_.to_output("cupy")
        self._mean_ = self._mean_.to_output("cupy")
        self._noise_variance_ = self._noise_variance_.to_output("cupy")
        self._singular_values_ = self._singular_values_.to_output("cupy")
        self._explained_variance_ = self._explained_variance_.to_output("cupy")
        self._explained_variance_ratio_ = \
            self._explained_variance_ratio_.to_output("cupy")


def _validate_sparse_input(X):
    """
    Validate the format and dtype of sparse inputs.
    This function throws an error for any cupy.sparse object that is not
    of type cupy.sparse.csr_matrix or cupy.sparse.csc_matrix.
    It also validates the dtype of the input to be 'float32' or 'float64'

    Parameters
    ----------
    X : scipy.sparse or cupy.sparse object
        A sparse input
    Returns
    -------
    X : The input converted to a cupy.sparse.csr_matrix object
    """

    acceptable_dtypes = ('float32', 'float64')

    # NOTE: We can include cupyx.scipy.sparse.csc.csc_matrix
    # once it supports indexing in cupy 8.0.0b5
    acceptable_cupy_sparse_formats = \
        (cp.sparse.csr_matrix)

    if X.dtype not in acceptable_dtypes:
        raise TypeError("Expected input to be of type float32 or float64."
                        " Received %s" % X.dtype)
    if scipy.sparse.issparse(X):
        return cp.sparse.csr_matrix(X)
    elif cp.sparse.issparse(X):
        if not isinstance(X, acceptable_cupy_sparse_formats):
            raise TypeError("Expected input to be of type"
                            " cupy.sparse.csr_matrix or"
                            " cupy.sparse.csc_matrix. Received %s"
                            % type(X))
        else:
            return X


def _gen_batches(n, batch_size, min_batch_size=0):
    """
    Generator to create slices containing batch_size elements, from 0 to n.
    The last slice may contain less than batch_size elements, when batch_size
    does not divide n.

    Parameters
    ----------
    n : int
    batch_size : int
        Number of element in each batch
    min_batch_size : int, default=0
        Minimum batch size to produce.
    Yields
    ------
    slice of batch_size elements
    """

    if not isinstance(batch_size, numbers.Integral):
        raise TypeError("gen_batches got batch_size=%s, must be an"
                        " integer" % batch_size)
    if batch_size <= 0:
        raise ValueError("gen_batches got batch_size=%s, must be"
                         " positive" % batch_size)
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        if end + min_batch_size > n:
            continue
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)


def _safe_accumulator_op(op, x, *args, **kwargs):
    """
    This function provides numpy accumulator functions with a float64 dtype
    when used on a floating point input. This prevents accumulator overflow on
    smaller floating point dtypes.
    Parameters
    ----------
    op : function
        A cupy accumulator function such as cp.mean or cp.sum
    x : cupy array
        A numpy array to apply the accumulator function
    *args : positional arguments
        Positional arguments passed to the accumulator function after the
        input x
    **kwargs : keyword arguments
        Keyword arguments passed to the accumulator function
    Returns
    -------
    result : The output of the accumulator function passed to this function
    """

    if cp.issubdtype(x.dtype, cp.floating) and x.dtype.itemsize < 8:
        result = op(x, *args, **kwargs, dtype=cp.float64).astype(cp.float32)
    else:
        result = op(x, *args, **kwargs)
    return result


def _incremental_mean_and_var(X, last_mean, last_variance, last_sample_count):
    """Calculate mean update and a Youngs and Cramer variance update.
    last_mean and last_variance are statistics computed at the last step by the
    function. Both must be initialized to 0.0. In case no scaling is required
    last_variance can be None. The mean is always required and returned because
    necessary for the calculation of the variance. last_n_samples_seen is the
    number of samples encountered until now.
    From the paper "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data to use for variance update
    last_mean : array-like, shape: (n_features,)
    last_variance : array-like, shape: (n_features,)
    last_sample_count : array-like, shape (n_features,)
    Returns
    -------
    updated_mean : array, shape (n_features,)
    updated_variance : array, shape (n_features,)
        If None, only mean is computed
    updated_sample_count : array, shape (n_features,)
    Notes
    -----
    NaNs are ignored during the algorithm.
    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247
    """

    # old = stats until now
    # new = the current increment
    # updated = the aggregated stats
    last_sum = last_mean * last_sample_count
    new_sum = _safe_accumulator_op(cp.nansum, X, axis=0)

    new_sample_count = cp.sum(~cp.isnan(X), axis=0)
    updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        new_unnormalized_variance = (
            _safe_accumulator_op(cp.nanvar, X, axis=0) * new_sample_count)
        last_unnormalized_variance = last_variance * last_sample_count

        # NOTE: The scikit-learn implementation has a np.errstate check
        # here for ignoring invalid divides. This is not implemented in
        # cupy as of 7.6.0
        last_over_new_count = last_sample_count / new_sample_count
        updated_unnormalized_variance = (
            last_unnormalized_variance + new_unnormalized_variance +
            last_over_new_count / updated_sample_count *
            (last_sum / last_over_new_count - new_sum) ** 2)

        zeros = last_sample_count == 0
        updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
        updated_variance = updated_unnormalized_variance / updated_sample_count

    return updated_mean, updated_variance, updated_sample_count


def _svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    Parameters
    ----------
    u : cupy.ndarray
        u and v are the output of `cupy.linalg.svd`
    v : cupy.ndarray
        u and v are the output of `cupy.linalg.svd`
    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.
    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = cp.argmax(cp.abs(u), axis=0)
        signs = cp.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, cp.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = cp.argmax(cp.abs(v), axis=1)
        signs = cp.sign(v[list(range(v.shape[0])), max_abs_rows])
        u *= signs
        v *= signs[:, cp.newaxis]
    return u, v
