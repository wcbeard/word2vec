import builtins
from collections import Counter, OrderedDict
from functools import wraps
from itertools import repeat, islice
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import numpy.random as nr
from pandas import Series, DataFrame, Index
from typing import Union, Iterable, Dict, List

import toolz.curried as z
from voluptuous import Any, Invalid


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
def cdist(v: '(n,)', M: '(n, m)', mnorms=None):
    "Cosine distance"
    if mnorms is None:
        mnorms = np.linalg.norm(M, axis=1)
    norms = np.linalg.norm(v) * mnorms
    return 1 - (v @ M.T) / norms


def get_closest(wd='death', W=None, Wnorm=None):
    if isinstance(wd, str):
        wvec = W.ix[wd]
    else:
        wvec = wd
    dst = cdist(wvec, W, mnorms=Wnorm)
    return dst.idxmin()


def get_closestn(wd='death', n=20, W=None, Wnorm=None, freq=None, exclude=[], just_word=False):
    if Wnorm is None:
        Wnorm = np.ones(len(W))
    if isinstance(wd, str):
        wvec = W.ix[wd]
    else:
        wvec = wd
    dst = cdist(wvec, W, mnorms=Wnorm)
    # if exclude:
    #     dst = dst[~W.index.isin(exclude)]
    if n == 1 and just_word:
        r = dst.idxmin()
        if r not in exclude:
            return Index([r])
    closests = dst.sort_values(ascending=True)[:n + len(exclude)]
    closests = closests[~closests.index.isin(exclude)][:n]
    if just_word:
        return closests.index
    cvecs = W.ix[closests.index]
    df = DataFrame(OrderedDict([  #  ('Freq', freq.ix[dv.wds(closests)]),
                                ('Dist', dst.ix[closests.index]),
                                ('Size', np.diag(cvecs @ cvecs.T)),
            ]))
    if freq is not None:
        df['Freq'] = closests.index.map(freq.get)
    df.Dist = df.Dist.map('{:.2f}'.format)
    df.Size = df.Size.map('{:.1f}'.format)
    # df.Freq = df.Freq.round()

    # get_closestn(W=w)
    return df.reset_index(drop=0).rename(columns={'index': 'Word'})


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