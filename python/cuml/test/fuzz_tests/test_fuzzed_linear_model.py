import traceback

import cupy as cp
import hypothesis.strategies as st
import numpy as np
import pytest

from hypothesis import given, settings
from hypothesis.extra.numpy import arrays as arrays_strategy
from hypothesis.extra.numpy import from_dtype as from_dtype_strategy
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.model_selection import train_test_split

from cuml import LinearRegression as cuLinearRegression
from cuml.test.utils import array_equal

@given(
    X=arrays_strategy(
        np.float64,
        (1000, 20),
        elements=from_dtype_strategy(
            np.dtype('float64'),
            allow_nan=False,
            allow_infinity=False,
            min_value = -1e5,
            max_value = 1e5
        )
    ),
    y=arrays_strategy(
        np.float64,
        (1000,),
        elements=from_dtype_strategy(
            np.dtype('float64'),
            allow_nan=False,
            allow_infinity=False,
            min_value = -1e5,
            max_value = 1e5
        )
    )
)
@settings(deadline=None, max_examples=50)
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
def test_linear_regression_model_default(X, y, datatype):
    X = X.astype(datatype)
    y = y.astype(datatype)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    # Initialization of cuML's linear regression model
    cuols = cuLinearRegression()

    # fit and predict cuml linear regression model
    cu_fail = False
    try:
        cuols.fit(X_train, y_train)
        cuols_predict = cuols.predict(X_test)
    except Exception:
        traceback.print_exc()
        cu_fail = True

    # sklearn linear regression model initialization and fit
    skols = skLinearRegression()

    sk_fail = False
    try:
        skols.fit(X_train, y_train)
        skols_predict = skols.predict(X_test)
    except Exception:
        traceback.print_exc()
        sk_fail = True

    if cu_fail:
        assert(sk_fail)
    else:
        assert array_equal(skols_predict, cuols_predict, 1e-1, with_sign=True)

