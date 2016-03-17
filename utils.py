from collections import namedtuple, OrderedDict, defaultdict
from functools import reduce, partial
from importlib import reload
from itertools import repeat, islice  # , filterfalse, tee
import numpy as np
import pandas as pd
from pandas import DataFrame
import operator as op
from os.path import join
import os
import re
import site
import toolz.curried as z
import time


# Text handling
Chapter = namedtuple('Chapter', 'num title text')
bookpat_re = re.compile(r'''\A(?P<title>.+)
\n*
(?:(?:.+\n+)+?)
(?P<body>
    (Chapter\ 1)
    \n+
    (.+\n*)+
)''', re.VERBOSE)

chappat_re = re.compile(r'''(Chapter (\d+)\n+((?:.+\n+)+))+''')
chapsep_re = re.compile(r'Chapter (\d+)\n(.+)\n+')


class Book(object):
    def __init__(self, title, chapters: {int: Chapter}):
        self.chapters = chapters
        self.title = title
        self.txts = OrderedDict()
        for n, chap in sorted(chapters.items()):
            setattr(self, 't{}'.format(n), chap.text)
            self.txts[n] = chap.text
        txt = reduce(op.add, self.txts.values())
        self.txt = clean_text(txt)


class BookSeries(object):
    def __init__(self, n=7):
        bks = {i: parsebook(i, vb=False) for i in range(1, n + 1)}

        self.txts = OrderedDict()
        for n, bk in sorted(bks.items()):
            setattr(self, 'b{}'.format(n), bk.txt)
            self.txts[n] = bk.txt
        txt = reduce(op.add, self.txts.values())
        self.txt = clean_text(txt)


def parsebook(fn="src/txt/hp1.txt", vb=False):
    p = print if vb else (lambda *x, **y: None)
    if isinstance(fn, int):
        fn = "src/txt/hp{}.txt".format(fn)
    p('Reading {}'.format(fn))
    with open(fn, 'rb') as f:
        txt = f.read().decode("utf-8-sig")

    gd = bookpat_re.search(txt).groupdict()

    booktitle = gd['title']
    body = gd['body']

    chs = chapsep_re.split(body)[1:]
    book = {int(chnum): Chapter(int(chnum), title, text) for chnum, title, text in z.partition(3, chs)}
    return Book(booktitle, book)


def clean_text(t):
    reps = {
        '’': "'",
        '‘': "'",
        '“': '"',
        '”': '"',
        '\xad': '',
        '—': '-',
       }

    def rep(s, frto):
        fr, to = frto
        return s.replace(fr, to)
    t = reduce(rep, reps.items(), t)
    return t


def mod_axis(df, f, axis=0):
    df = df.copy()
    if not axis:
        df.index = f(df.index)
    else:
        df.columns = f(df.columns)
    return df


def pvalue(x, xs, side=4):
    "side: 1=>low p-value, 2=>hi, 3=>min(lo,hi), 4=> min(lo,hi) * 2"
    l = np.sum(x <= np.array(xs))
    r = np.sum(x >= np.array(xs))
    np1 = len(xs) + 1
    lp = (1 + l) / np1
    rp = (1 + r) / np1

    if side == 1: return lp
    elif side == 2: return rp
    elif side == 3: return min(lp, rp)
    elif side == 4: return min(lp, rp) * 2
    else: raise ValueError('`side` arg should be ∈ 1..4')


# Graph
def dedupe_wrd_repr(s):
    d = {}
    dfd = defaultdict(int)
    for tok in s:
        dfd[tok.orth_] += 1
        n = dfd[tok.orth_]
        # print(tok.i, tok, n)
        if n > 1:
            d['{}[{}]'.format(tok.orth_, n)] = tok.i
        else:
            d[tok.orth_] = tok.i
    return {v: k for k, v in d.items()}

def add_edge(src, dst, G, reprdct=None):
    """Since this is a tree, append an underscore for duplicate
    destination nodes"""
    G.add_edge(reprdct[src.i], reprdct[dst.i])

def add_int_edge(src, dst, G, **_):
    G.add_edge(src.i, dst.i)


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


# def partition(pred, iterable):
#     'https://docs.python.org/dev/library/itertools.html#itertools-recipes'
#     t1, t2 = tee(iterable)
#     return filterfalse(pred, t1), filter(pred, t2)


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


# Gensim fast/slow mode
[sdir] = site.getsitepackages()
gensim_dir = join(sdir, 'gensim/models')
fast_file = join(gensim_dir, 'word2vec_fast.py')
slow_file = join(gensim_dir, 'word2vec_slow.py')
wv_file = join(gensim_dir, 'word2vec.py')


def mk_gensim_sym(slow=False):
    dst = slow_file if slow else fast_file
    if os.path.islink(wv_file):
        os.remove(wv_file)

    elif os.path.exists(wv_file):
        raise Exception('{} not a symlink!'.format(wv_file))
    os.symlink(dst, wv_file)
    #print(dst)
    return True


def reset_gensim(slow=False, gensim=None):
    mk_gensim_sym(slow=slow)
    reload(gensim.models.word2vec)


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