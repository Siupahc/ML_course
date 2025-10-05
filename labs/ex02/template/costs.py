# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss(y, tx, w, method="MSE"):
    """Calculate the loss using either MSE or MAE or RMSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx.dot(w)
    if method == "MSE":
        return 0.5 * np.mean(e**2)
    elif method == "MAE":
        return np.mean(np.abs(e))
    elif method == "RMSE":
        return np.sqrt(2 * compute_loss(y, tx, w, method="MSE"))
    else:
        raise ValueError("Unknown method: {}".format(method))