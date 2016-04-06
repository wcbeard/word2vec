import builtins
from collections import Counter, OrderedDict
from functools import wraps
from itertools import repeat, islice, count
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import numpy.random as nr
from numpy.linalg import norm
from operator import itemgetter as itg
from pandas import Series, DataFrame, Index
from typing import Union, Iterable, Dict, List
from numba import jit

import toolz.curried as z
from voluptuous import Any, Invalid, Schema, ALLOW_EXTRA
import numba_utils as nbu
import utils as ut

nopython = jit(nopython=True)

map = z.comp(list, builtins.map)
UNKNOWN = '<UNK>'


# Matrix weight representation
class Cat(object):
    "Join and split W matrices for passing as single arg"
    @staticmethod
    def join(w1, w2):
        return np.hstack([w1, w2.T])

    @staticmethod
    def split(Wall):
        n = Wall.shape[1] / 2
        W1, W2_ = Wall[:, :n], Wall[:, n:]
        return W1, W2_.T


def init_w(V, N, seed=None, test=False):
    if test:
        a = np.arange(V * N).reshape(V, N)
        m = np.hstack([a, a])
        return DataFrame(m) if test == 2 else m
    nr.seed(seed)
    W1_ = (nr.rand(V, N) - .5) / N  #sp.sparse.csr_matrix()
    W2_ = (nr.rand(N, V) - .5) / N  #sp.sparse.csr_matrix()
    return Cat.join(W1_, W2_)


class WordVectorizer(DictVectorizer):
    """Given string, convert to count dict of count 1.

    """
    def fit(self, X, y=None, transform=True):
        if transform:
            X = map(self.todict, X)
            if y:
                y = map(self.todict, y)
        return super().fit(list(X) + [self.todict(UNKNOWN)], y=y)

    def get_(self, wd: str):
        try:
            return self.vocabulary_[wd]
        except IndexError:
            return self.vocabulary_[UNKNOWN]

    def get(self, wds):
        if isinstance(wds, str):
            return self.get_(wds)
        return map(self.get_, wds)

    def wds(self, ids):
        vc = self.get_feature_names()
        return map(vc.__getitem__, ids)

    def transform(self, X: Union[str, List[str]], y=None):
        dct_unk = z.compose(self.to_unk, self.todict)
        if isinstance(X, str):
            x = dct_unk(X)
        else:
            x = map(dct_unk, X)
        return super().transform(x, y=y)

    def to_unk(self, dct: Dict[str, int]) -> Dict[str, int]:
        missing = dct.keys() - self.vocabulary_.keys()
        if missing:
            dct[UNKNOWN] = dct.get(UNKNOWN, 0)
            for m in missing:
                dct[UNKNOWN] += dct.pop(m)
        return dct

    def inverse_transform(self, X, dict_type=dict, transform=True):
        ret = super().inverse_transform(X, dict_type=dict_type)
        if not transform:
            return ret
        return map(self.fromdict, [ret])[0]

    @staticmethod
    def todict(s: Union[str, Iterable[str]]) -> Dict[str, int]:
        if isinstance(s, str):
            return {s: 1}
        return dict(zip(s, repeat(1)))

    @staticmethod
    def fromdict(ds: Iterable[Dict[str, int]]) -> List[str]:
        "[{a: 1}, {b: 1}] -> [a, b]"
        assert all(len(d) == 1 for d in ds), ds
        return [next(iter(d)) for d in ds]


# Sliding windows
def get_window(lst: List[str], C:int=4):
    for i, wd in enumerate(lst[C:-C], C):
        st = i - C
        end = i + C
        yield lst[st:i], wd, lst[i + 1:end + 1]


def get_win(corp: [str], i: int, C: int=4) -> (str, [str]):
    """Get `C` preceding and `C` following words, along with word i
    i: C..L-C
    """
    L = len(corp)
    assert i < L - C, 'Must be followed by {} words'.format(C)
    assert i >= C, 'Must be preceded by {} words'.format(C)
    # corp[i - C:i], corp[i], corp[i + 1:i+C+1]
    if isinstance(corp, list):
        return corp[i], corp[i - C:i] + corp[i + 1:i+C+1]
    else:
        return corp[i], np.append(corp[i - C:i], corp[i + 1:i+C+1])



def get_rand_wins(corp, C=4, seed=None, n=None):
    if seed is not None:
        nr.seed(seed)
    L = len(corp)
    ilo = C
    ihi = L - C

    def get_rand_wins_():
        while 1:
            i = nr.randint(ilo, ihi)
            yield get_win(corp, i, C=C)

    if n is not None:
        return islice(get_rand_wins_(), n)
    return get_rand_wins_()


def get_wins(i, corpus, winlen=4, cat=True):
    "get_wins(2, [0, 1, 2, 3, 4], winlen=2, cat=0) == ([0, 1], [3, 4])"
    lix = max(i - winlen, 0)
    rix = min(i + 1 + winlen, len(corpus))
    lwin, rwin = corpus[lix:i], corpus[i+1:rix]
    if cat:
        return list(lwin) + list(rwin)
    return lwin, rwin


def inspect_freq_thresh(txt: [str]):
    """Find `thresh` for subsampling. Choose it to be the `Freq` for
    the least frequent word to be decreased.
    """
    vcs = Series(Counter(txt)).sort_values(ascending=0)

    freqdf = ((vcs.sort_values(ascending=True).cumsum()
               / vcs.sum() * 100).reset_index(drop=0)
              .rename(columns={'index': 'Word'}))  # .round(1)
    freqdf['Count'] = freqdf.Word.map(vcs.get)
    freqdf['Freq'] = freqdf.Count / freqdf.Count.sum()
    freqdf = freqdf.sort_values('Count', ascending=False).reset_index(drop=1)
    return freqdf


# Closest word

def combine(plus=[], minus=[], W=None, wd2row=None):
    ixs = np.array([wd2row[p] for p in plus + minus])
    wa = W.values if isinstance(W, DataFrame) else W
    return nbu.ix_combine_(wa, ixs, len(plus))


def top_matches(sim, wds, n, skip=[]):
    sk = set(skip)
    N = n + len(sk)
    ixs = np.argpartition(sim, -N)[-N:]
    res = sim[ixs]
    sim_, wds_ = sim[ixs], wds[ixs]
    return sorted([(s, w) for s, w in zip(sim_, wds_) if w not in sk], key=itg(0), reverse=1)[:n]


def cos_sim2(a, b, bnorm=None):
    dp = a @ b
    bnorm = nbu.norm_jit2d(b) if bnorm is None else bnorm
    return dp / (nbu.norm_jit1d(a) * bnorm)


def closest(plus=[], minus=[], W=None, wds=None, wd2row=None, bnorm=None, n=1):
    combined = combine(plus=plus, minus=minus, W=W, wd2row=wd2row)
    # _B = wn.values.T
    # _bn = norm_jit2d(_B)
    sims = cos_sim2(combined, W.T, bnorm=bnorm)
    res = top_matches(sims, wds, n, skip=plus + minus)
    if n > 1:
        return res
    [(score, match)] = res
    return match


class Eval(object):
    def __init__(self, qs_loc='src/questions-words.txt'):
        self.qs_loc = qs_loc
        self.load_qs()
        self.wn = None
        self.wnorm = None

    def load_qs(self, qs_loc=None):
        with open(qs_loc or self.qs_loc, 'r') as f:
            qlns = f.read().splitlines()
        sections, qs = ut.partition(lambda s: not s.startswith(':'), qlns)
        self.qs = map(str.split, qs)

    def norm_w(self, W):
        self.wn = np.divide(W, norm(W, axis=1)[:, None])
        assert np.allclose(norm(self.wn, axis=1), 1)
        self.wnorm = np.linalg.norm(self.wn)
        self.vocab = np.array(self.wn.index)
        self.vocabs = set(self.vocab)
        self.wd2ix = dict(zip(self.vocab, count()))
        self._kwds = dict(W=self.wn, wd2row=self.wd2ix, bnorm=self.wnorm, wds=self.vocab, n=1)
        return self.wn

    def score_(self, qs=None):
        qs = self.qs if qs is None else qs
        assert self.wn is not None, 'Call `norm_w(W)`'
        res = [((closest(plus=[b, c], minus=[a], **self._kwds), ans), (a, b, c))
               for a, b, c, ans in qs
              if not {a, b, c} - self.vocabs]
        self.res = res
        return sum(a == b for (a, b), _ in res)

    def score(self, W, vocab=None, qs=None):
        if vocab is not None:
            W = DataFrame(W, index=vocab)
        self.norm_w(W)
        return self.score_(qs=qs)


def score_wv(w, vocab):
    evl = Eval()
    evl.norm_w(DataFrame(w, index=vocab))
    return evl.score_()


# Subsample
def get_subsample_prob(txt, thresh=.001):
    cts = Counter(txt)
    freq = np.array([cts[w] for w in txt]) / sum(cts.values())
    p = 1 - np.sqrt(thresh / freq)
    return np.clip(p, 0, 1)


def get_subsample(txt, thresh=.001) -> (['keep'], ['drop']):
    """
    Drop words with frequency above given threshold according to frequency.
    From "Distributed Representations of Words and Phrases and their Compositionality"
    Returns pair of (left in words, left out words)
    """
    p = get_subsample_prob(txt, thresh=thresh)
    drop = np.zeros_like(p, dtype=bool)

    for pval in sorted(set(p[p > 0]), reverse=1):
        bm = p == pval
        n = bm.sum()
        pdrop = nr.random(n) < pval
        drop[bm] = pdrop

    print('Dropping {:.2%} of words'.format(drop.mean()))
    return txt[~drop], txt[drop]


# Config helpers
def orig_type(f):
    return wraps(f)(lambda x: type(x)(f(x)))


def update(dct, **kw):
    d2 = dct.copy()
    d2.update(**kw)
    return d2


def even(x):
    if x % 2:
        raise Invalid('x must be an even number')
    return x

Num = Any(float, int)
Dict = lambda x: Any(x, {})

Conf = Schema(dict(
    eta=Num,      # initial learning rate
    min_eta=Num,  # final learning rate
    # norm=Num,
    accumsec=Num, # total training time so far
    # norms=Dict({int: float}),
    # gradnorms=Dict({int: float}),
    N=int,       # input (== output) vector dimensions
    K=int,       # number of negative samples drawn for each true context word
    term={},     # terminating condition:  iters=#words, secs=# secs or empty for full epoch
    iter=int,    # if `term` not empty, this keeps track the index of the last word,
                 # to pick up from on the next iteration
    epoch=int,   # Number of epochs so far
    dir=str,     # directory to save CSV of gradient to
    C=even,      # window diameter; must be an even number
    thresh=Num,  # gradient norm threshold for decreasing learning rate
    pad=Num,
), extra=ALLOW_EXTRA, required=True)
Conf = orig_type(Conf)
