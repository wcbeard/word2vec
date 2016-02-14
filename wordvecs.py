
# coding: utf-8

# In[ ]:

get_ipython().run_cell_magic('javascript', '', "var csc = IPython.keyboard_manager.command_shortcuts\ncsc.add_shortcut('Ctrl-k','ipython.move-selected-cell-up')\ncsc.add_shortcut('Ctrl-j','ipython.move-selected-cell-down')\ncsc.add_shortcut('Shift-m','ipython.merge-selected-cell-with-cell-after')")


# In[ ]:

from project_imports import *

get_ipython().magic('matplotlib inline')

cachedir = 'cache/'
memory = Memory(cachedir=cachedir, verbose=0)


# In[ ]:

import time
import utils as ut; reload(ut);


# In[ ]:

bksall = ut.BookSeries(7)
all_text = bksall.txt


# - [Word-Vectors](#Word-Vectors)

# In[ ]:

from spacy.en import English
get_ipython().magic('time nlp = English()')


# In[ ]:

get_ipython().run_cell_magic('time', '', 'bktksall = {i: nlp(bktxt, tag=True, parse=True, entity=True)\n            for i, bktxt in bksall.txts.items()}')


# In[ ]:

get_ipython().magic('time atks = nlp(bksall.txt)')


# # Get named entity phrases

# phrase_ents = Series(Counter([e.orth_ for e in ltoks.ents if e.label in ent_nums and ' ' in e.orth_])).sort_values(ascending=0)
# ent_cands = phrase_ents[phrase_ents > 2].sort_index()
# ent_cands

# ent_cnt = Series(Counter([(e.label_, e.orth_) for e in atks.ents]))
# ent_cnt.reset_index(drop=0).to_csv('/tmp/ents.csv')

# In[ ]:

ent_lab2num = {e.label_: e.label for e in atks.ents}
ent_nums = sorted({ent_lab2num[lab] for lab in 'EVENT FAC GPE ORG PERSON WORK_OF_ART'.split()})


# %%time
# phrase_ents_low = (Series(Counter(
#             [e.orth_.lower() for e in atks.ents if e.label in ent_nums
#              and ' ' in e.orth_])).sort_values(ascending=0))

# In[ ]:

get_ipython().run_cell_magic('time', '', "phrase_ents = (Series(Counter(\n            [e.orth_ for e in atks.ents if e.label in ent_nums\n             and ' ' in e.orth_])).sort_values(ascending=0)).reset_index(drop=0)\nphrase_ents.columns = ['Phrase', 'Count']")


# phrase_ents2 = (Series(Counter(
#             [e.orth_ for e in atks.ents if e.label in ent_nums
#              and ' ' in e.orth_])).sort_values(ascending=0)).reset_index(drop=0)

# In[ ]:

def rep_phrase(s):
    """Pass string as a phrase to replace with words joined by
    underscores. If returns None, then don't replace it. Else,
    the returned string is ok to replace.
    >>> rep_phrase('the Harry Potter') == 'Harry Potter'
    """
    if isinstance(s, str):
        wds = s.split()
    else:
        wds = s
    if len(wds) < 2:
        return
    if wds[0][0].islower():
        return rep_phrase(wds[1:])
    if wds[-1][0].islower():
        return rep_phrase(wds[:-1])
    return ' '.join(wds)
    
    
def clean_hyphen(s):
    "Keep hyphens that separate words, discard rest"
    surr_hyph_re = re.compile(r"(?<=[A-Za-z])-(?=[A-Za-z])")
    sentinel = b'\uF5DC'.decode('unicode_escape')
    s = surr_hyph_re.sub(sentinel, s)
    s = re.sub('-', '', s)
    s = re.sub(sentinel, '-', s)
    return s


def tokenize(s: str=bksall.txt):
    split_re = re.compile(r"[^A-Za-z-\d_']")
    return np.array(filter(bool, [t.strip("'") for t in split_re.split(clean_hyphen(s))]))


def multi_replace(pairs, text):
    return reduce(lambda accum, x: accum.replace(x[0], x[1]),  pairs, text)


def multi_replace(dct, text):
    """Replace occurrence of keys in `text` with corresponding values,
    longest keys first"""
    pairs = sorted(dct.items(), key=lambda x: len(x[0]), reverse=True)
    return reduce(lambda accum, x: accum.replace(x[0], x[1]), pairs, text)

# test_rep_phrase()


# In[ ]:

def test_rep_phrase():
    assert rep_phrase('Harry Potter') == 'Harry Potter'
    assert rep_phrase('the Harry Potter') == 'Harry Potter'
    assert rep_phrase('Harry Potter.') == 'Harry Potter.'
    assert rep_phrase("Harry Potter's toy") == "Harry Potter's"
    assert rep_phrase("the Potter") is None
    
def test_tokenize():
    assert clean_hyphen('whoops-a-daisy') == 'whoops-a-daisy'
    assert clean_hyphen('whoops- a-daisy') == 'whoops a-daisy'
    assert clean_hyphen('whoops-a -daisy') == 'whoops-a daisy'
    assert clean_hyphen('whoops-a -daisy') == 'whoops-a daisy'
    assert clean_hyphen(' -a a- -a- a-a a-a-a ') == ' a a a a-a a-a-a '
    assert all(tokenize('whoops-a-daisy, -there- -it goes-') == ['whoops-a-daisy', 'there', 'it', 'goes'])
    
test_rep_phrase()
test_tokenize()


# In[ ]:

phrases = (phrase_ents.assign(Clean=lambda x: x.Phrase.map(clean_hyphen))
           .query('Count >= 7 & Clean == Phrase').Phrase)  # .map(rep_phrase).dropna()
phrase2wds = OrderedDict([(w, ut.phrase2wd(w)) for w in phrases])


# extras2 = [h for i, h in enumerate(hss2) if h not in s1]
# print(len(extras2))
# extras2
# extras1 = [h for i, h in enumerate(hss) if h not in s2]
# print(len(extras1))
# extras1
# a2 = clean_hyphen(all_text)
# hs2 = hyph_re.findall(a2)
# hss2 = filter(surr_hyph_re.findall, hs2)
# 
# phrase_ents

# phrase_ents
# phrase_ents[phrase_ents.Phrase.str.startswith('Pro')]
# n = 1040000
# ' '.join(toks[n:n+1000])
# n = 5009000
# print(phrased_text[n:n+9500])
# phrase_ents.value_counts(normalize=0)
# 
# phrase_ents[phrase_ents ]

# # Get Bigram Phrases

# In[ ]:

# def get_big_scores(big_dct, uni_dct, δ=2):
#     global w1, w2
#     return {(w1, w2): (cts - δ) / ((uni_dct[w1] or print('w1', w1)) *
#                          (uni_dct[w2] or print('w2', w2)))
#             for w1, w1dct in big_dct.items()
#             for w2, cts in w1dct.items()}

def get_big_scores(big_dct, uni_dct, δ=2):
    #global w1, w2
    dct = {}
    for w1, w1dct in big_dct.items():
        for w2, cts in w1dct.items():
            assert uni_dct[w1] or uni_dct[w2], ("Some word isn't "
                                                "accounted for in unigrams")
            dct[(w1, w2)] = (cts - δ) / (uni_dct[w1] * uni_dct[w2])
    return dct

def get_bigrams(toks, δ=9):
    bigd = defaultdict(lambda: defaultdict(int))
    for w1, w2 in builtins.zip(toks, toks[1:]):
        if (w1 == 'va') or (w2 == 'va'):
            print('w1', w1)
            print('w2', w2)
        bigd[w1][w2] += 1

    unid = defaultdict(int)
    unid.update({k: sum(v.values()) for k, v in bigd.items()}.items() | {toks[-1]: 1}.items())
    
    big_scores = get_big_scores(bigd, unid, δ=δ)
    
    bigdf = (Series(big_scores).reset_index(drop=0).sort_values(0, ascending=0)
     .rename(columns={'level_0': 'W1', 'level_1': 'W2', 0: 'Score'})
     .reset_index(drop=1)
    )
    
    minerva = bigdf.query('W1 == "Professor" & W2 == "McGonagall"').Score.iloc[0]
    bigdf = bigdf.query('Score >= @minerva')
    
    bigdf['From'] = bigdf.W1 + ' ' + bigdf.W2
    bigdf['From'] = bigdf.W1 + '_' + bigdf.W2
    return bigdf[bigdf.W1.map(iscap) & bigdf.W2.map(iscap)
                 & ~bigdf.W1.str.isupper() & ~bigdf.W2.str.isupper()]
#                  & (bigdf.W1.str.len() > 1) & (bigdf.W2.str.len() > 1)]

def filter_phrases(bigrams: {' ': '_'}, phrases: {' ': '_'}):
    phrases_ = set(phrases.values())
    is_subset = lambda x: x not in phrases_ and not any(x in ent for ent in phrases_)
    return z.valfilter(is_subset, bigrams)


iscap = lambda x: x[0].isupper()


# In[ ]:

bigdf = get_bigrams(tokenize(all_text), δ=9)
bigram_rep = dict(zip(bigdf.W1 + ' ' + bigdf.W2, bigdf.W1 + '_' + bigdf.W2))
bigram_rep = filter_phrases(bigram_rep, phrase2wds)
bigram_rep.update({
        w: '_'.join(w.split()) for w in
        'Mrs Norris;Every Flavor Beans;Expecto Patronum;Boy Who Lived'.split(';')})
bigram_rep = z.keymap(lambda x: x.replace('Mrs ', 'Mrs. '), bigram_rep)
punct_replace = {"St Mungo's": "St. Mungo's"}
bigram_rep = z.keymap(lambda x: punct_replace.get(x, x), bigram_rep)
bigram_rep = z.keyfilter(lambda x: x not in 'Every Flavor;Flavor Beans;Boy Who;Who Lived'.split(';'),
                         bigram_rep )


# ## Phrasify
# 
# revdict = lambda d: {v: k for k, v in d.items()}
# rbigram_rep = revdict(bigram_rep)
# rphrase2wds = revdict(phrase2wds)

# In[ ]:

with open('src/name2phrase.txt', 'r') as f:
    name2phrase = dict(map(str.split, f.read().splitlines()))


# In[ ]:

# %%time 
to_replace = z.merge(phrase2wds, bigram_rep)
multicase_phrases = ut.get_multi_case(OrderedDict(sorted(to_replace.items(), key=lambda x: -len(x[0]))), all_text)
to_replace.update(multicase_phrases)
phrased_text = multi_replace(to_replace, all_text)
# phrased_text = multi_replace(phrase2wds.items(), all_text)
toks = tokenize(phrased_text)
toks = np.array([name2phrase.get(t, t) for t in toks])


# In[ ]:

toksl = {t.lower() for t in toks}
diffs = {t.lower() for t in z.valmap(lambda t: name2phrase.get(t, t), to_replace).values()} - toksl
diffs


# toks = tokenize(phrased_text)
# 
# for t in toks:
#     if 'Vernon_Dursley' in t:
#         print(t)

# casedat = (DataFrame([(k, len(set(ut.find_all(all_text, k))),
#                        len(set(ut.find_all(lt, k.lower()))))
#                       for k, v in to_replace.items()], columns=['Word', 'Case', 'Nocase'])
#            .query('Case != Nocase').assign(Ratio = lambda x: x.eval('Case / Nocase'))
#            .sort_values('Ratio', ascending=True).reset_index(drop=1)
#           )
# 
#     # casedat['Ratio'] = casedat.eval('Case / Nocase')

# # Word Vectors

# In[ ]:

import scipy as sp
sp.sparse.csr_matrix.__matmul__ = sp.sparse.csr_matrix.dot
# import sklearn

# import autograd.numpy as np
# from autograd import grad
import numpy as np
# %load_ext line_profiler


# In[ ]:

import wordvec_utils as wut; reload(wut);
import utils as ut; reload(ut);
import test; reload(test);
from wordvec_utils import Cat, WordVectorizer, get_rand_wins, get_wins
from voluptuous import Schema


# ## Subsample

# In[ ]:

# THRESH = 0.0014
THRESH = 0.15

def get_subsample_prob(txt, thresh=.001):
    cts = Counter(txt)
    freq = np.array([cts[w] for w in txt]) / sum(cts.values())
    p = 1 - np.sqrt(thresh / freq)
    return np.clip(p, 0, 1)


def get_subsample(txt, thresh=.001):
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


# threshs = wut.inspect_freq_thresh(toks).sort_values('Freq').reset_index(drop=1)
# [ix] = threshs.Freq.sort_values().searchsorted(THRESH)
# threshs.iloc[ix-5:ix+5]
# 
# threshs[-3:]

# ## Objective functions
# 
# The equation for the skip-gram objective function is the following
# \begin{align}
# E & = -\log \prod_{c=1} ^{C}
#     \frac {\exp (u_{c,j^*_c})}
#           {\sum_{j'=1} ^ V \exp(u_{j'})} \\
#   & = -\sum^C_{c=1} u_{j^*_c} + C \cdot \log \sum ^ V _{j'=1} \exp(u_j')
# \end{align}
# and implemented below as `vanilla_likelihood`:

# In[ ]:

def vanilla_likelihood(wi, cwds, dv=None):
    wix = dv.get(wi)
    cixs = dv.get(cwds)
    C = len(cwds)

    def logloss(Wall):
        W1, W2 = Cat.split(Wall)
        h = W1[wix, :]  # ∈ ℝⁿ
        u = np.dot(h, W2)  # u[1083] == 427  ∈ ℝⱽ
        ucs = u[cixs]
        return -np.sum(ucs) + C * np.log(np.sum(np.exp(u)))
    return logloss


# After reading about the negative sampling modification, however, I replaced the usage of the above with `ns_obj`, written later.
# 
# ### Negative sampling
# #### Unigram model

# In[ ]:

from numba import jit
# from autograd.numpy import exp, log
from numpy import exp, log
from builtins import zip as izip, range as xrange

nopython = jit(nopython=True)


# In[ ]:

@nopython
def bisect_left(a, v):
    """Based on bisect module at (commit 1fe0fd9f)
    cpython/blob/master/Modules%2F_bisectmodule.c#L150 
    """
    lo, hi = 0, len(a)
    while (lo < hi):
        mid = (lo + hi) // 2
        if a[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    return lo


# In[ ]:

def unigram(txt, pow=.75):
    "Unigram^(3/4) model"
    cts = Series(Counter(txt))
    
    # If txt is integers, fill in missing values (likely for unknown token)
    # with 0 probability to reliably use index to identify token
    int_txt = cts.index.dtype == int
    if int_txt:
        missing_tokens = set(range(cts.index.max())) - set(cts.index)
        for msg in missing_tokens:
            cts.loc[msg] = 0
        cts = cts.sort_index()
        
    N = len(txt)
    ctsdf = ((cts / cts.sum()) ** pow).reset_index(drop=0)
    ctsdf.columns = ['Word', 'Prob']
    if int_txt:
        assert (ctsdf.Word == ctsdf.index).all()
    return ctsdf




# In[ ]:

def neg_sampler_pd(xs, K, pow=.75):
    ug = unigram(xs, pow=pow)
    for seed in count():
        yield ug.Word.sample(n=K, weights=ug.Prob, random_state=seed, replace=True)
        
        
def neg_sampler_np(xs, K, cache_len=1000, use_seed=False, pow=.75):
    "Faster neg. sampler without the pandas overhead"
    ug = unigram(xs, pow=pow)
    p = ug.Prob.values / ug.Prob.sum()
    a = ug.Word.values

    for seed in count():
        if use_seed:
            nr.seed(seed)
        Wds = nr.choice(a, size=(cache_len, K), p=p)
        for wds in Wds:
            yield wds
            
            
def neg_sampler_np_l(xs, K, cache_len=1000, pow=.75):
    "Faster neg. sampler without the pandas overhead"
    ug = unigram(xs, pow=pow)
    p = ug.Prob.values / ug.Prob.sum()
    a = list(ug.Word.values)

    @nopython
    def sample_():
        while 1:
            Wds = nr.choice(a, size=(cache_len, K), p=p)
            for i in xrange(len(Wds)):
                yield Wds[i]
    return sample_


def neg_sampler_j(xs, K, pow=.75):
    ug = unigram(xs, pow=pow)
    cum_prob = ug.Prob.cumsum() / ug.Prob.sum()
    return neg_sampler_j_(cum_prob.values, K)

@nopython
def neg_sampler_j_(cum_prob, K):
    while 1:
        l = []
        for i in xrange(K):
            l.append(bisect_left(cum_prob, nr.rand()))
        yield l


# gen = sample_(ug.Cum_prob.values, 8)

# ### Check distributions

# In[ ]:

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
toks = le.fit_transform(all_text.split())


# In[ ]:

get_ipython().magic('timeit genj = neg_sampler_j(toks, 8)')
get_ipython().magic('timeit gennp = neg_sampler_np(toks, 8)')


# In[ ]:

gennp = neg_sampler_np(toks, 8)
get_ipython().magic('lprun -s -f neg_sampler_np next(gennp)')


# In[ ]:

get_ipython().magic('lprun -s -f unigram neg_sampler_j(toks, 8)')


# In[ ]:

genj = neg_sampler_j(toks, 8)
gennp = neg_sampler_np(toks, 8)
genp = neg_sampler_pd(toks, 8)

next(genj); next(gennp); next(genp);


# In[ ]:

n = 100000
get_ipython().magic('time csp = Series(Counter(x for xs in it.islice(genp, n // 100) for x in xs))')
get_ipython().magic('time csnp = Series(Counter(x for xs in it.islice(gennp, n) for x in xs))')
get_ipython().magic('time csj = Series(Counter(x for xs in it.islice(genj, n) for x in xs))')

ug = unigram(toks, pow=.75)
cts = DataFrame({'Numba': csj, 'Numpy': csnp, 'Pandas': csp}).fillna(0)
probs = cts / cts.sum()
probs['Probs'] = ug.Prob / ug.Prob.sum()


# In[ ]:

def plot_dist(xcol=None, subplt=None):
    plt.subplot(subplt)
    probs.plot(x=xcol, y='Probs', ax=plt.gca(), kind='scatter', alpha=.25)
    _, xi = plt.xlim(None)
    _, yi = plt.ylim(0, None)
    end = min(xi, yi)
    plt.plot([0, end], [0, end], alpha=.2)
    
plt.figure(figsize=(16, 10))
plot_dist(xcol='Numba', subplt=131)
plot_dist(xcol='Numpy', subplt=132)
plot_dist(xcol='Pandas', subplt=133)


# # Train
# ## Gradient

# list(sliding_window(ls[:10], C=6))

# %timeit list(sliding_window(ls, C=4))
# %timeit list(sliding_window2(ls, C=4))
# %timeit list(sliding_window3(ls, C=4))

# $$
# E = -\log \sigma(\boldsymbol v_{w_O}' ^T \boldsymbol h)
#     - \sum^K _{i=1} \log \sigma (-\boldsymbol v_{w_i}' ^T \boldsymbol h)
# $$
# 
# $$
#     \frac{\partial E}
#          {\partial \boldsymbol v_{w_j}' ^T \boldsymbol h}
#          = \sigma(\boldsymbol v_{w_j}' ^T \boldsymbol h) -t_j
# $$

# In[ ]:

# @nopython
def sig(x):
    return 1 / (1 + np.exp(-x))

# take = z.compose(list, it.islice)


# In[ ]:

getNall = lambda W: W.shape[1] // 2

@ut.memoize
def mrange(*a):
    return list(xrange(*a))

def get_vecs1(Wall, w_ix: int=0, vo_ix: [int]=1, negsamp_ixs: [int]=None):
    if negsamp_ixs is None:
        negsamp_ixs = mrange(2, len(Wall))
    N = getNall(Wall)
    h = Wall[w_ix, :N]  # ∈ ℝⁿ
    vwo = Wall[vo_ix, N:]
    negsamps = Wall[negsamp_ixs, N:]
    return h, vwo, negsamps


def gen_labels(negsamps):
    return [1] + [0] * len(negsamps)


def ns_loss_grad_dot(h=None, vout=None, label=None):
    return sig(vout @ h) - label


def ns_loss_grads(h: 'v[n]', vout: '[v[n]]', label: 'v[n]'):
    dot = ns_loss_grad_dot(h=h, vout=vout, label=label)
    return dot * vout, dot * h


def zeros(shape, z=ut.memoize(lambda shape: np.zeros(shape))):
    return z(shape).copy()

    
def ns_grad(Wsub):
    # global hgrad, vgrad, Wsub, N
#     h, vwo, negsamps = get_vecs1jit(Wsub)
    h, vwo, negsamps = get_vecs1(Wsub)
    N = getNall(Wsub)
    Wsub_grad = zeros(Wsub.shape)
    
    for i, vout, label in izip(count(1), it.chain([vwo], negsamps), gen_labels(negsamps)):
        hgrad, vgrad = ns_loss_grads(h, vout, label)
        Wsub_grad[0, :N] += hgrad
        Wsub_grad[i, N:] += vgrad

    return Wsub_grad


# ns_grad(Wsub)
# %timeit ns_grad(Wsub)

# ## Gradient check
# The following gradient checking functionality based on [the UFLDL tutorial](http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/) can be used to ensure that autograd is working as expected. It may be redundant with autograd calculating everything automatically, but I felt better checking manually for a few iterations.

# In[ ]:

from typing import Callable


def ns_loss(h, vwo, vwi_negs):
    """This should be called on the subset of the matrix (Win || Wout')
    determined by row indices `wi, win_ix, negwds`.
    Indexing relevant rows before passing to `logloss` seems to speed up autograd.
    """
    # Win, Wout = Cat.split(Wall_sub)  # copy
    # return -np.log(σ(np.dot(vwo.T, h))) - np.sum(np.log(σ(-np.dot(vwi_negs.T, h))))
    # return -np.log(sig(vwo @ h)) - np.sum(np.log(sig(-vwi_negs @ h)))
    negsum = 0
    for j in xrange(len(vwi_negs)):
        negsum += np.log(sig(-vwi_negs[j] @ h))
        
    return -np.log(sig(vwo @ h)) - negsum


def ns_loss_vec(h, vwo, vwi_negs):
    """This should be called on the subset of the matrix (Win || Wout')
    determined by row indices `wi, win_ix, negwds`.
    Indexing relevant rows before passing to `logloss` seems to speed up autograd.
    """
    # Win, Wout = Cat.split(Wall_sub)  # copy
    # return -np.log(σ(np.dot(vwo.T, h))) - np.sum(np.log(σ(-np.dot(vwi_negs.T, h))))
    # return -np.log(sig(vwo @ h)) - np.sum(np.log(sig(-vwi_negs @ h)))
    return -np.log(sig(vwo @ h)) - np.sum(np.log(sig(-vnegs @ h )))

# vnegs = Wout[:, neg_samps].T.copy()
# ns_loss_jit = jit(nopython=1)(ns_loss)
# ns_loss_vec_jit = jit(nopython=1)(ns_loss_vec)
# a1 = ns_loss(h, vwo, vnegs)
# a2 = ns_loss_jit(h, vwo, vnegs)
# a3 = ns_loss_vec(h, vwo, vnegs)
# a4 = ns_loss_vec_jit(h, vwo, vnegs)
# assert np.isclose(a1, a2) and np.isclose(a1, a3) and np.isclose(a1, a4)
# ns_loss = ns_loss_vec_jit


# In[ ]:

def J(Wsub):
    N = getNall(Wsub)
    h, vwo, vwi_negs = get_vecs1(Wsub)
    # h, vwo, vwi_negs = Wsub[0, :N], Wsub[1, N:], Wsub[range(2, len(Wsub)), N:]
    return ns_loss(h, vwo, vwi_negs)

def check_grad_(W, i: int=None, j: int=None, eps=1e-6, J: Callable=None):
    "From eqn at http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/"
    Wneg, Wpos = W.copy(), W.copy()
    Wneg[i, j] -= eps
    Wpos[i, j] += eps
    return (J(Wpos) - J(Wneg)) / (2 * eps)

def approx_grad(W, J=J):
    n, m = W.shape
    grad = np.zeros_like(W)
    for i in range(n):
        for j in range(m):
            grad[i, j] = check_grad_(W, i=i, j=j, eps=1e-6, J=J)
    return grad


# DataFrame(approx_grad(Wsub))
# J(Wsub)


# %timeit ns_loss(h, vwo, vnegs)
# %timeit ns_loss_vec(h, vwo, vnegs)
# %timeit ns_loss_jit(h, vwo, vnegs)
# %timeit ns_loss_vec_jit(h, vwo, vnegs)

# In[ ]:

import negsamp_grad; reload(negsamp_grad);
from negsamp_grad import ns_grad as ns_grad_jit


# In[ ]:

W = wut.init_w(1000, 50, seed=1)
Wsub = W[:8]
W.shape
# ns_grad(Wsub)

assert np.allclose(ns_grad(Wsub), ns_grad_jit(Wsub))
assert np.allclose(approx_grad(Wsub), ns_grad_jit(Wsub))

get_ipython().magic('timeit approx_grad(Wsub)')
get_ipython().magic('timeit ns_grad(Wsub)')
get_ipython().magic('timeit ns_grad_jit(Wsub)')


# ## SGD

# In[ ]:

from wordvec_utils import Dict, Num, even, orig_type, update


# In[ ]:

import utils as ut; reload(ut);
from voluptuous import ALLOW_EXTRA
from collections import deque
import os

def ping():
    get_ipython().system('say done')


# In[ ]:

Conf = Schema(dict(
        λ=Num,
        norm=Num,  accumsec=Num, norms=Dict({int: float}),  gradnorms=Dict({int: float}),
        N=int, K=int, term={}, iter=int, epoch=int, dir=str,
        C=even,  # full window size; must be an even number
        thresh=Num,  # gradient norm threshold for decreasing learning rate
), extra=ALLOW_EXTRA, required=True)
Conf = orig_type(Conf)


# In[ ]:

cnf = ut.AttrDict(
    λ=.5, norm=0, accumsec=0, norms={}, gradnorms={}, N=100,
    C=4, K=6, iter=0, thresh=15, epoch=0,
    term=dict(iters=None,
              secs=10
    ),
    dir='cache',
)
cnf = Conf(cnf)


# ### Sliding window

# In[ ]:

def sliding_window(xs, C=4, start_pos=0):
    """Iterates through corpus, yielding input word
    and surrounding context words"""
    #assert isinstance(xs, list)
    winsize = C // 2
    N = len(xs)
    for i, x in enumerate(xs, start_pos):
        ix1 = max(0, i-winsize)
        ix2 = min(N, i+winsize+1)
        yield x, xs[ix1:i] + xs[i + 1:ix2]


@nopython
def bounds_check_window(i, xs, winsize, N):
    x = xs[i]
    ix1 = max(0, i-winsize)
    ix2 = min(N, i+winsize+1)
    return x, xs[ix1:i] + xs[i + 1:ix2]


@nopython
def sliding_window_jit(xs, C=4):
    """Iterates through corpus, yielding input word
    and surrounding context words"""
    winsize = C // 2
    N = len(xs)
    for i in xrange(winsize):
        yield bounds_check_window(i, xs, winsize, N)
    for i in xrange(winsize, N-winsize):
        context = []
        for j in xrange(i-winsize, i+winsize+1):
            if j != i:
                context.append(xs[j])
        yield xs[i], context  # xs[i-winsize:i] + xs[i + 1:i+winsize+1]
    for i in xrange(N-winsize, N):
        yield bounds_check_window(i, xs, winsize, N)


# In[ ]:

samp_toks = list(nr.randint(0, 1e6, size=100005))
list(sliding_window_jit(samp_toks[:100]))
run_window = lambda f: list(f(samp_toks))

get_ipython().magic('timeit run_window(sliding_window)')
get_ipython().magic('timeit run_window(sliding_window_jit)')
assert run_window(sliding_window) == run_window(sliding_window_jit)


# In[ ]:

#         if i % 1000 == 0:
#             _gn = gradnorms[i + gradnormax] = np.linalg.norm(grad)
#             maxnorms.append(_gn)
#             norms[i + normax] = np.linalg.norm(W)
#             if max(maxnorms) > cf.thresh:
#                 cf['λ'] *= 2 / 3
#                 print('Setting λ: {:.2f}'.format(cf['λ']))
#                 maxnorms.clear()
# #             else:
# #                 print('{:.1f}'.format(max(maxnorms)), end='; ')
#         # sys.stdout.flush()


# In[ ]:

@nopython
def grad_norm(Wsub):
    """Calculate norm of gradient, where first row
    is input vector, rest are output vectors. For any row,
    half of the entries are zeros, which allows a lot of
    skipping for a faster computation"""
    n = Wsub.shape[1] // 2
    sm = 0
    for i in xrange(n):
        sm += Wsub[0, i] ** 2
    for i in xrange(1, len(Wsub)):
        for j in xrange(n, 2 * n):
            sm += Wsub[i, j] ** 2
    return np.sqrt(sm)


# In[ ]:

grd = ns_grad_jit(Wsub)
assert np.isclose(grad_norm(grd), np.linalg.norm(grad_norm(grd)))

get_ipython().magic('timeit np.linalg.norm(grad_norm(grd))')
get_ipython().magic('timeit grad_norm(grd)')


# In[ ]:

get_ipython().magic('load_ext line_profiler')


# In[ ]:

get_ipython().magic('pinfo z.partition')


# In[ ]:




# In[ ]:

def sgd(W=None, corp=None, cf={}, ns_grad=ns_grad, neg_sampler=None):
    # TODO: ensure neg samp != wi
    if not os.path.exists(cf.dir):
        os.mkdir(cf.dir)
    st = time.time(); cf = Conf(cf)  #.copy()
    norms = dict(cf.norms); gradnorms = dict(cf.gradnorms)
    assert cf.N == W.shape[1] / 2, 'shape of W disagrees with conf'
    maxnorms = deque([], 5)

    normax = max(norms or [0]); gradnormax = max(gradnorms or [0]);
    # global Win, Wout, w, cont, negsamp_lst, c, negsamps, sub_ixs
    # Win, Wout = Cat.split(W)
    iter_corpus = corp[cf.iter:]
    learning_rates = np.linspace(cf['λ'], cf['λ'] * .1, len(iter_corpus))
    if neg_sampler is None: neg_sampler = neg_sampler_np(corp, cf.K)    
    iters_ = izip(count(cf.iter),
                  sliding_window(iter_corpus, C=cf.C),
                  z.partition(cf.C, neg_sampler),
                  learning_rates,
                 )
    iters = ut.timeloop(iters_, **cf.term)

    for i, (w, cont_), negsamp_lst, eta in iters:
        cont = [x for x in cont_ if x != w] if w in cont_ else cont_
        for c, negsamps_ in zip(cont, negsamp_lst):
            negsamps = ([x for x in negsamps_ if x not in {w, c}]
                        if set([w, c]) & set(negsamps_) else negsamps_)
            sub_ixs = [w, c] + negsamps # list(negsamps)
            Wsub = W[sub_ixs]
            grad = ns_grad(Wsub)
            gnorm = grad_norm(grad)
            
            if gnorm > 5:  # clip gradient
                grad /= gnorm
            W[sub_ixs] -= eta * grad    
                
        if i % 1000 == 0 and np.isnan(grad).any():
            global grd
            print('ruh roh!'); grd = grad
            return

    tdur = time.time() - st
    print('{:.2f} mins'.format(tdur / 60))
    cf2 = update(cf, norms=norms, gradnorms=gradnorms, iter=i+1)
    cf2['accumsec'] += tdur
    if not cf2.term:
        DataFrame(W, index=vc).to_csv(os.path.join(cf2.dir, 'n{}_e{}.csv'.format(cf.N, cf.epoch)))
        cf2['epoch'] += 1
        cf2 = update(cf2, iter=0)
    else:
        print(i, 'iters')
    # ping()
    return W, cf2


# In[ ]:

ngsamp = neg_sampler_j(toki, cnfe.K)
kw = dict(ns_grad=ns_grad_jit, neg_sampler=ngsamp)
get_ipython().magic("lprun -T lp5.txt -s -f sgd sgd(W=We.copy(), corp=toki, cf=update(cnfe, term={'iters': 10000}), **kw) # ls[:20]")


# In[ ]:

get_ipython().magic("lprun -T lp.txt -s -f sgd sgd(W=We.copy(), corp=toki, cf=update(cnfe, term={'iters': 10000}), ns_grad=ns_grad) # ls[:20]")


# rand_ixs = lambda W, n=8, axis=0: nr.randint(0, W.shape[axis], size=n)

# In[ ]:

@nopython
def ineg(W, sub_ixs, grad):
    # for i in xrange(len(grad)):
    for i, _ in enumerate(grad):
        W[sub_ixs[i]] -= grad[i]
        
def inegp(W, sub_ixs, grad):
    W[sub_ixs] -= grad


# In[ ]:

def test_ineg1(wsub_):
    wsub = wsub_.copy()
    wsub -= gr * .1
    return wsub

def test_ineg2(wsub_):
    wsub = wsub_.copy()
    ineg(wsub, gr)
    wsub -= gr * .1
    return wsub


# In[ ]:

ixs = [ 1, 4, 7, 99, 486, 263, 924]
w1, w2 = wtst.copy(), wtst.copy()
# w1, w2 = wtst[ixs].copy(), wtst[ixs].copy()
gr = ns_grad(w1[ixs])


# In[ ]:

get_ipython().magic('timeit ineg(w1, ixs, gr)')
get_ipython().magic('timeit inegp(w2, ixs, gr)')


# In[ ]:

cnfe = update(cnf, C=4, iter=0, term=dict(), N=100, λ=.5, dir='cache/v12', epoch=0)

# tks = np.array(all_text.split())
stoks, dropped = get_subsample(toks, thresh=THRESH)
# assert 'Albus_Dumbledore' in stoks
dv = WordVectorizer().fit(stoks)
toki = [dv.vocabulary_[x] for x in stoks]
vc = dv.feature_names_
W = wut.init_w(len(vc), cnfe.N, seed=1)
with open('txt.txt','w') as f:
    f.write('\n'.join(stoks))
    
We = W.copy()

get_ipython().system('say done')


# In[ ]:

np.allclose(w1, w2)


# In[ ]:

ineg(w1, ixs, gr)


# In[ ]:

inegp(w2, ixs, gr)
np.allclose(w1, w2)


# In[ ]:

wtst[ixs]


# In[ ]:

wtst = wut.init_w(1000, 50)
wsub = wtst[:8].copy()
gr = ns_grad(wsub)
# gr


# In[ ]:

test_ineg1(wsub)


# In[ ]:




# In[ ]:

wsub.shape


# In[ ]:

grad


# In[ ]:

weight_normj(grad)


# In[ ]:

weight_norm(grad), weight_norm2(grad), weight_normj(grad)


# In[ ]:

weight_norm_jit(grad), np.linalg.norm(grad)


# In[ ]:

get_ipython().magic('timeit weight_norm(grad)')


# In[ ]:

get_ipython().magic('timeit weight_norm2(grad)')


# In[ ]:

get_ipython().magic('timeit weight_norm_jit(grad)')


# In[ ]:

get_ipython().magic('timeit weight_normj(grad)')


# In[ ]:

get_ipython().magic('timeit np.linalg.norm(grad)')


# In[ ]:




# In[ ]:

grad, grad2 = grd, grd2
Wsub = W2[[w, c] + list(negsamps)]


# In[ ]:

DataFrame(approx_grad(Wsub))


# In[ ]:

DataFrame(Wsub)


# In[ ]:

DataFrame(ns_grad(W2[[w, c] + list(negsamps)]))


# In[ ]:

DataFrame(grad)


# In[ ]:

for i in range(20):
    print('Epoch {}'.format(cnfe.epoch))
    We2, cnfe = sgd(W=We.copy(), corp=toki, cf=update(cnfe, term=dict()))
    break


# In[ ]:




# In[ ]:

vc[w]


# In[ ]:

map(vc.__getitem__, [c])


# In[ ]:

map(vc.__getitem__, negsamps)


# In[ ]:

negsamps


# In[ ]:

del w, cont, c, negsamps


# In[ ]:

np.linalg.norm(grad, axis=1)


# In[ ]:

vc[19276]


# In[ ]:

toki[]


# In[ ]:

sgdpart = partial(sgd, W=W.copy(), corp=toki, cf=update(cnfe, term={'iters': 1}))
_ = sgdpart()


# In[ ]:




# In[ ]:

1


# In[ ]:




# In[ ]:

gradnorm_vec = np.linalg.norm(grad, axis=1)


# In[ ]:

if np.linalg.norm(gradnorm_vec) > 5:
    grad = grad / gradnorm_vec[:, None]


# In[ ]:

# gradnormed = np.divide(grad, gradnorm)
gradnormed = grad / gradnorm[:, None]


# In[ ]:

np.linalg.norm(gradnormed, axis=1)


# In[ ]:

gradnormed.sum(axis=1)


# In[ ]:

7.12
  1085718
6.00742
-> 684739


# In[ ]:

sgd(W=W.copy(), corp=toki, cf=update(cnf, term={'iters': 1}));
get_ipython().magic("prun -qD prof.txt sgd(W=W.copy(), corp=toki, cf=update(cnf, term={'iters': 5000})) # ls[:20]")


# In[ ]:




# In[ ]:




# We = pd.read_csv('cache/v9/n15_e26.csv', index_col=0).values

# In[ ]:

grd


# In[ ]:

plt.scatter(*zip(*gns), alpha=.1)


# In[ ]:

for i in range(20):
    print('Epoch {}'.format(cnfe.epoch))
    We2, cnfe = sgd(W=We.copy(), corp=toki, cf=update(cnfe, term=dict()))
    break


# In[ ]:

We2 - We


# In[ ]:

get_ipython().run_cell_magic('time', '', "W2, cnf2 = sgd(W=W.copy(), corp=toki, cf=update(cnf, term={'mins': 60})) # ls[:20]\nsw = sliding_window(toki, C=cnf.C)\n# 799509 iters\n# CPU times: user 12min 29s, sys: 4.47 s, total: 12min 34s\n# Wall time: 12min 35s")


# In[ ]:

get_ipython().run_cell_magic('time', '', 'W3, cnf3 = sgd(W=W2.copy(), corp=toki, cf=update(cnf2, iter=0)) # ls[:20]')


# %%time
# W3, cnf2 = sgd(W=W2.copy(), corp=toki, cf=update(cnf, term={'mins': 5})) # ls[:20]

# In[ ]:

ping()


# In[ ]:

W3, cnf3 = sgd(W=W3.copy(), corp=toki, cf=update(cnf3, term={'mins': 5}), ) # ls[:20]


# In[ ]:

wd = DataFrame(W2, index=vc)
wd.to_csv('cache/v2/n6_e1.csv')


# In[ ]:

wd.mean(axis=1).sort_values(ascending=True)


# In[ ]:

wd[10:200]


# In[ ]:

Series(cnfe.norms).plot()


# In[ ]:

for i, x in enumerate([-1, 1,6,8]):
    print(i, x)


# In[ ]:

s


# In[ ]:

cnf2.norms[48000]


# In[ ]:

plt.figure(figsize=(30, 10))
gnrms = Series(cnfe.gradnorms)
gnrms.plot()
pd.rolling_mean(gnrms, 20).plot()


# ## Update equation
# $$
#     \frac{\partial E}
#          {\partial \boldsymbol v_{w_j}' ^T \boldsymbol h}
#          = \sigma(\boldsymbol v_{w_j}' ^T \boldsymbol h) -t_j
# $$

# # Analyze word vectors

# In[ ]:



