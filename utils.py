from builtins import filter as ifilter
from functools import partial
from itertools import repeat, islice  # , filterfalse, tee
import itertools as it
import pandas as pd
from pandas import DataFrame
import re
import sys
import toolz.curried as z
import time


def mod_axis(df, f, axis=0):
    df = df.copy()
    if not axis:
        df.index = f(df.index)
    else:
        df.columns = f(df.columns)
    return df


def timeloop(it, secs=None, mins=None, iters=None):
    if mins is not None:
        secs = mins * 60
    secs = secs or float('inf')
    iters = iters or float('inf')
    start = time.time()
    for i, x in enumerate(it, 1):
        yield x
        if (i >= iters) or (time.time() - start > secs):
            raise StopIteration


class AttrDict(dict):
    "http://stackoverflow.com/a/14620633/386279"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        return type(self)(super().copy())


def test_print(vb):
    return print if vb else lambda *x, **_: None


def side_by_side(*ds, names=None):
    nms = iter(names) if names else repeat(None)
    dmultis = [side_by_side1(d, ctr=i, name=next(nms)) for i, d in enumerate(ds)]
    return pd.concat(dmultis, axis=1)


def side_by_side1(d, ctr=1, name=None):
    d = pd.DataFrame(d.copy())
    d.columns = pd.MultiIndex.from_product([[name or ctr], list(d)])
    return d


def memoize1(f):
    "http://code.activestate.com/recipes/578231-probably-the-fastest-memoization-decorator-in-the-/"
    class memodict(dict):
        __slots__ = ()
        def __missing__(self, key):
            print('m')
            print(self)
            self[key] = ret = f(key)
            return ret
    return memodict().__getitem__


def memoize(f):
    class memodict(dict):
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)

        def __missing__(self, key):
            ret = self[key] = f(*key)
            return ret

    return memodict().__getitem__


def partition(pred, iterable):
    'Use a predicate to partition entries into false entries and true entries'
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = it.tee(iterable)
    return it.filterfalse(pred, t1), ifilter(pred, t2)


def spr(*a, **k):
    print(*a, **k)
    sys.stdout.flush()


# Phrase searching functions
def find_all(st, substr, start_pos=0, accum=[]):
    "Return all indices of `st` where `subtr` starts"
    ix = st.find(substr, start_pos)
    if ix == -1:
        return accum
    return find_all(st, substr, start_pos=ix + 1, accum=accum + [ix])


def findall_ignore_case(substr_, ret_ixs=False, low_txt=None, txt=None):
    """Get indexes or words in `low_txt` that match `substr`, ignoring
    the case.
    Ensure that `substr` is surrounded by boundary chars.
    """
    assert txt is not None, "Must pass txt parameter"
    low_txt = low_txt or txt.lower()
    substr = substr_.lower()
    l = len(substr)
    ixs = find_all(low_txt, substr)
    if ret_ixs:
        return ixs
    pat_nocase = re.compile(r'\b{}'.format(substr))
    caseless_ix = find_all(low_txt, substr)

    return [txt[i:i+l] for i in caseless_ix if pat_nocase.search(low_txt[i-1:i+l+1])]


def get_multi_case(ks, txt, thresh=.9):
    """For each key `ks`, see if there exist multiple copies
    of that key in different cases. Ignore keys where at least
    `thresh` percent of all occurrences have the same case. For
    other keys where at least 10% of the instances have different
    cases, return a dict mapping those instances to the original
    key in `ks`"""
    lowtxt = txt.lower()
    find_icase = partial(findall_ignore_case, low_txt=lowtxt, txt=txt)
    casedata = (DataFrame([(k, len(find_all(txt, k)), len(find_icase(k)))
                for k in ks], columns=['Word', 'Case', 'Nocase'])
        .query('Case != Nocase').assign(Ratio=lambda x: x.eval('Case / Nocase'))
        .sort_values('Ratio', ascending=True).reset_index(drop=1)
    )
    return {diffcase: phrase2wd(k) for k in casedata.query('Ratio < @thresh').Word
        for diffcase in set(find_icase(k))}


# Gensim
def to_gensim_params(cnf, **kw):
    gparams = dict(
        size=cnf.N, # 80, #
        alpha=cnf.eta,
        min_alpha=cnf.min_eta,
        window=cnf.C / 2,
        sample=0,
        negative=cnf.K,  #[5, 7, 10, 12, 15, 17], 0
        sg=1,
        # iter=4,
    )
    gparams.update( **kw)
    return gparams


phrase2wd = lambda x: '_'.join(re.split(r'[ -]', x))
take = z.comp(list, islice)
ilen = lambda xs: sum(1 for _ in xs)