from collections import namedtuple, Counter, defaultdict, OrderedDict
from functools import wraps, partial
from glob import glob
from itertools import count
import itertools as it
import operator as op
from operator import itemgetter as itg, attrgetter as prop, methodcaller as mc
import os
from os.path import join
import re
import sys
import time
import warnings; warnings.filterwarnings("ignore")


from joblib import Parallel, delayed, Memory
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
import pandas as pd
from scipy import stats
import seaborn as sns
import toolz.curried as z

from IPython.display import Image

import builtins
from functools import wraps, reduce
from importlib import reload


def listify(f):
    @wraps(f)
    def wrapper(*a, **k):
        return list(f(*a, **k))
    return wrapper


map = listify(builtins.map)
range = listify(builtins.range)
filter = listify(builtins.filter)
zip = listify(builtins.zip)

imap = builtins.map
xrange = builtins.range
ifilter = builtins.filter
izip = builtins.zip

Series.__matmul__ = Series.dot
DataFrame.__matmul__ = DataFrame.dot

pd.options.display.notebook_repr_html = False
pd.options.display.width = 120