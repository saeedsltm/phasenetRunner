#!/usr/bin/env python3

from yaml import SafeLoader, load
from numpy import ma
import os
import sys


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
