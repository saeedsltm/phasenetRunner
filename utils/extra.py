#!/usr/bin/env python3

from yaml import SafeLoader, load
from numpy import ma, average, sqrt, where, logspace, append
import os
import sys
from sklearn.preprocessing import MinMaxScaler

def readConfiguration():
    if not os.path.exists("config.yml"):
        msg = "+++ Could not find configuration file! Aborting ..."
        print(msg)
        sys.exit()
    with open("config.yml") as f:
        config = load(f, Loader=SafeLoader)
    msg = "+++ Configuration file was loaded successfully."
    print(msg)
    return config


def handle_masked_arr(st):
    for tr in st:
        if isinstance(tr.data, ma.masked_array):
            tr.data = tr.data.filled()
    return st


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- NumPy ndarrays with the same shape.
    """
    avg = average(values, weights=weights)
    # Fast and numerically precise:
    variance = average((values-avg)**2, weights=weights)
    return (avg, sqrt(variance))


def weightMapper(weights, minW=0, reverse=False):
    if not reverse:
        sc = MinMaxScaler(feature_range=(minW, 1.0))
        W = logspace(minW, 1.0, 5)[::-1]
        W = sc.fit_transform(W.reshape(-1, 1))
        W = append(W, 0)
        for w1, w2, w in zip(W[:-1], W[1:], range(5)):
            weights = where((weights > w2) & (weights <= w1), w, weights)
    else:
        weights = where(weights == 0, 1.00, weights)
        weights = where(weights == 1, 0.75, weights)
        weights = where(weights == 2, 0.50, weights)
        weights = where(weights == 3, 0.25, weights)
        weights = where(weights == 4, 0.00, weights)
    return weights
