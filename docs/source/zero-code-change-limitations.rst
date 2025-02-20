Known Limitations
-----------------

General Limitations
~~~~~~~~~~~~~~~~~~~

TODO(wphicks): Fill this in
TODO(wphicks): Pickle

Algorithm-Specific Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO(wphicks): Fills these in. Document when each will fall back to CPU, how to
assess equivalence with CPU implementations, and significant differences in
algorithm, as well as any other known issues.


``sklearn.cluster.KMeans``
^^^^^^^^^^^^^^^^^^^^^^^^^^

The default initialization algorithm used by ``cuml.accel`` is similar, but different.
``cuml.accel`` uses the ``"scalable-k-means++"`` algorithm, for more details refer to
:class:`cuml.KMeans`.

This means that the ``cluster_centers_`` attribute will not be exactly the same as for
the scikit-learn implementation. The ID of each cluster (``labels_`` attribute) might
change, this means samples labelled to be in cluster zero for scikit-learn might be
labelled to be in cluster one for ``cuml.accel``. The ``inertia_`` attribute might
differ as well if different cluster centers are used. The algorithm might converge
in a different number of iterations, this means the ``n_iter_`` attribute might differ.

To check that the resulting trained estimator is equivalent to the scikit-learn
estimator, you can evaluate the similarity of the clustering result on samples
not used to train the estimator. Both ``adjusted_rand_score`` and ``adjusted_mutual_info_score``
give a single score that should be above ``0.9``. For low dimensional data you
can also visually inspect the resulting cluster assignments.

``cuml.accel`` will not fall back to scikit-learn.


``sklearn.cluster.DBSCAN``
^^^^^^^^^^^^^^^^^^^^^^^^^^

``sklearn.decomposition.PCA``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sklearn.decomposition.TruncatedSVD``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sklearn.ensemble.RandomForestClassifier`` / ``sklearn.ensemble.RandomForestRegressor``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The random forest in ``cuml.accel`` uses a different algorithm to find tree node splits.
When choosing split thresholds, the ``cuml.accel`` random forest considers only quantiles
as threshold candidates, whereas the scikit-learn random forest considers all possible
feature values from the training data. As a result, the ``cuml.accel`` random forest
may choose different split thresholds from the scikit-learn counterpart, leading to
different tree structure. You can tune the fineness of the quantiles by adjusting the
``n_bins`` parameter.

Some parameters have limited support:
* ``max_samples`` must be float, not integer.

The following parameters are not supported:
* ``min_weight_fraction_leaf``
* ``monotonic_cst``
* ``ccp_alpha``
* ``class_weight``
* ``warm_start``
* ``oob_score``

TODO(hcho3): Add the list of supported ``criterion``, once the PR for mapping ``split_criterion``
to ``criterion`` lands.

TODO(hcho3): If the PR mapping ``max_leaves`` to ``max_leaf_nodes`` lands, add explanation about
the behavior of ``max_leaf_nodes``. The cuML RF treats this parameter as a "soft constraint".

``sklearn.kernel_ridge.KernelRidge``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sklearn.linear_model.LinearRegression``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sklearn.linear_model.LogisticRegression``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sklearn.linear_model.ElasticNet``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sklearn.linear_model.Ridge``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sklearn.linear_model.Lasso``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sklearn.manifold.TSNE``
^^^^^^^^^^^^^^^^^^^^^^^^^

``sklearn.neighbors.NearestNeighbors``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Algorithm Limitations:
    * The "kd_tree" and "ball_tree" algorithms are not implemented in CUDA. When specified, the implementation will automatically fall back to using the "brute" force algorithm.

* Distance Metrics:
    * Only Minkowski-family metrics (euclidean, manhattan, minkowski) and cosine similarity are GPU-accelerated
    * Not all metrics are supported for algorithms.
    * The "mahalanobis" metric is not supported on GPU and will trigger a fallback to CPU implementation.
    * The "nan_euclidean" metric for handling missing values is not supported on GPU.
    * Custom metric functions (callable metrics) are not supported on GPU.

* Other Limitations:
    * Only the "uniform" weighting strategy is supported. Other weighting schemes will cause fallback to CPU
    * The "radius" parameter for radius-based neighbor searches is not implemented and will be ignored

``sklearn.neighbors.KNeighborsClassifier``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Algorithm Limitations:
    * The "kd_tree" and "ball_tree" algorithms are not implemented in CUDA. When specified, the implementation will automatically fall back to using the "brute" force algorithm.

* Distance Metrics:
    * Only Minkowski-family metrics (euclidean, manhattan, minkowski) and cosine similarity are GPU-accelerated
    * Not all metrics are supported for algorithms.
    * The "mahalanobis" metric is not supported on GPU and will trigger a fallback to CPU implementation.
    * The "nan_euclidean" metric for handling missing values is not supported on GPU.
    * Custom metric functions (callable metrics) are not supported on GPU.

* Other Limitations:
    * Only the "uniform" weighting strategy is supported for vote counting.
    * Distance-based weights ("distance" option) will trigger CPU fallback.
    * Custom weight functions are not supported on GPU.

``sklearn.neighbors.KNeighborsRegressor``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Algorithm Limitations:
    * The "kd_tree" and "ball_tree" algorithms are not implemented in CUDA. When specified, the implementation will automatically fall back to using the "brute" force algorithm.

* Distance Metrics:
    * Only Minkowski-family metrics (euclidean, manhattan, minkowski) and cosine similarity are GPU-accelerated
    * Not all metrics are supported for algorithms.
    * The "mahalanobis" metric is not supported on GPU and will trigger a fallback to CPU implementation.
    * The "nan_euclidean" metric for handling missing values is not supported on GPU.
    * Custom metric functions (callable metrics) are not supported on GPU.

* Regression-Specific Limitations:
    * Only the "uniform" weighting strategy is supported for prediction averaging.
    * Distance-based prediction weights ("distance" option) will trigger CPU fallback.
    * Custom weight functions are not supported on GPU.

``umap.UMAP``
^^^^^^^^^^^^^

``hdbscan.HDBSCAN``
^^^^^^^^^^^^^^^^^^^
