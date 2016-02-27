
# coding: utf-8

# In[ ]:

get_ipython().run_cell_magic('javascript', '', "var csc = IPython.keyboard_manager.command_shortcuts\ncsc.add_shortcut('Ctrl-k','ipython.move-selected-cell-up')\ncsc.add_shortcut('Ctrl-j','ipython.move-selected-cell-down')\ncsc.add_shortcut('Shift-m','ipython.merge-selected-cell-with-cell-after')")


# In[ ]:

from project_imports import *
import utils as ut; reload(ut);
get_ipython().magic('matplotlib inline')


# In[ ]:

# import scipy as sp
# sp.sparse.csr_matrix.__matmul__ = sp.sparse.csr_matrix.dot
# import numpy as np


# # Word Vectors

# In[ ]:

import wordvec_utils as wut; reload(wut);
from wordvec_utils import Cat, WordVectorizer
from voluptuous import Schema, ALLOW_EXTRA


# ## Subsample

# In[ ]:

THRESH = 0.15

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

def ping():
    get_ipython().system('say done')


# In[ ]:

Conf = Schema(dict(
        eta=Num, min_eta=Num,
        norm=Num,  accumsec=Num, norms=Dict({int: float}),  gradnorms=Dict({int: float}),
        N=int, K=int, term={}, iter=int, epoch=int, dir=str,
        C=even,  # full window size; must be an even number
        thresh=Num,  # gradient norm threshold for decreasing learning rate
), extra=ALLOW_EXTRA, required=True)
Conf = orig_type(Conf)


# In[ ]:

cnf = ut.AttrDict(
    eta=.1, min_eta=.0001, norm=0, accumsec=0, norms={}, gradnorms={}, N=100,
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
def bounds_check_window(i, xs: [int], winsize, N):
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
      
@nopython
def concat(a, b):
    na = len(a)
    n = na + len(b)
    c = np.empty(n, dtype=a.dtype)
    for i in xrange(na):
        c[i] = a[i]
    for i in xrange(len(b)):
        c[i + na] = b[i]
    return c

@nopython
def bounds_check_window_arr(i, xs: np.array, winsize, N):
    x = xs[i]
    ix1 = max(0, i-winsize)
    ix2 = min(N, i+winsize+1)
    return x, concat(xs[ix1:i], xs[i + 1:ix2])

@nopython
def sliding_window_jit_arr(xs, C=4):
    """Iterates through corpus, yielding input word
    and surrounding context words"""
    winsize = C // 2
    N = len(xs)
    for i in xrange(winsize):
        yield bounds_check_window_arr(i, xs, winsize, N)
    for i in xrange(winsize, N-winsize):
        context = np.empty(C, dtype=np.int64)
        for ci in xrange(winsize):
            context[ci] = xs[i - winsize + ci]
            context[winsize + ci] = xs[i + 1 + ci]
#             if j != i:
#                 context[ci] = xs[i - winsize + ci]
        yield xs[i], context  # xs[i-winsize:i] + xs[i + 1:i+winsize+1]
    for i in xrange(N-winsize, N):
        yield bounds_check_window_arr(i, xs, winsize, N)


# bounds_check_window(1, toks, 4, len(toks))
# bounds_check_window_arr(0, toks, 4, len(toks))
# bounds_check_window(i, xs, winsize, N)
# 
# s = sliding_window_jit_arr(samp_toks)

# In[ ]:

samp_toks = nr.randint(0, 1e6, size=100005)
samp_toksl = list(samp_toks)
list(sliding_window_jit(samp_toksl[:100]))
run_window = lambda f, toks=samp_toksl: list(f(toks))

get_ipython().magic('timeit run_window(sliding_window)')
get_ipython().magic('timeit run_window(sliding_window_jit)')
get_ipython().magic('timeit run_window(sliding_window_jit_arr, toks=samp_toks)')
# assert run_window(sliding_window) == run_window(sliding_window_jit) == run_window(sliding_window_jit_arr)


# ### Norm

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


# ### Sgd

# In[ ]:

def sgd(W=None, corp=None, cf={}, ns_grad=ns_grad, neg_sampler=None, vc=None, sliding_window=sliding_window):
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
    learning_rates = np.linspace(cf.eta, cf.min_eta, len(iter_corpus))
    assert neg_sampler is not None, "Give me a negative sampler!"
    iters_ = izip(count(cf.iter),
                  sliding_window(iter_corpus, C=cf.C),
                  z.partition(cf.C, neg_sampler),
                  learning_rates,
                 )
    iters = ut.timeloop(iters_, **cf.term)

    for i, (w, cont_), negsamp_lst, eta in iters:
        cont = [x for x in cont_ if x != w] if w in cont_ else cont_
        for c, negsamps in zip(cont, negsamp_lst):
            if set([w, c]) & set(negsamps):
                negsamps = [x for x in negsamps if x not in {w, c}]

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
fast_opts = dict(ns_grad=ns_grad_jit, neg_sampler=ngsamp, sliding_window=sliding_window_jit)
get_ipython().magic("lprun -T lp5.txt -s -f sgd sgd(W=We.copy(), corp=toki, cf=update(cnfe, term={'iters': 10000}), **fast_opts) # ls[:20]")


# In[ ]:

sgd(W=We.copy(), corp=toki, cf=update(cnfe, term={'iters': 10000}), **kw)


# In[ ]:

get_ipython().magic("lprun -T lp.txt -s -f sgd sgd(W=We.copy(), corp=toki, cf=update(cnfe, term={'iters': 10000})) # ls[:20]")


# rand_ixs = lambda W, n=8, axis=0: nr.randint(0, W.shape[axis], size=n)

# In[ ]:

ixs = [ 1, 4, 7, 99, 486, 263, 924]
w1, w2 = wtst.copy(), wtst.copy()
# w1, w2 = wtst[ixs].copy(), wtst[ixs].copy()
gr = ns_grad(w1[ixs])


# In[ ]:

get_ipython().magic('timeit ineg(w1, ixs, gr)')
get_ipython().magic('timeit inegp(w2, ixs, gr)')


# In[ ]:

cnfe = update(cnf, C=4, iter=0, term=dict(), N=100,  dir='cache/v12', epoch=0)


# In[ ]:


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

brown.s


# In[ ]:

for i in range(20):
    print('Epoch {}'.format(cnfe.epoch))
    We2, cnfe = sgd(W=We.copy(), corp=toki, cf=update(cnfe, term=dict()), **fast_opts)
    break
    


# ## Corpus

# In[ ]:

import nltk; reload(nltk)
from nltk.corpus import brown, reuters
from gensim.models import Word2Vec


# In[ ]:

cnfe = update(cnf, C=4, iter=0, term=dict(), N=100, dir='cache/v12', epoch=0)


# In[ ]:

default

size=100
alpha=0.025
window=5
sample=0
negative=0
sg=1
iter=4

min_alpha=0.0001


# In[ ]:

gparams = dict(
    size=120, # 80, #
    alpha=0.025,
    window=2,
    sample=0,
    negative=2,  #[5, 7, 10, 12, 15, 17], 0
    sg=1,
    iter=4,
)


# In[ ]:

def ev(mod):
    ans = mod.accuracy('src/questions-words.txt', restrict_vocab=10000)
    sect = [d for d in ans if d['section'] == 'total']
    return sum([1 for d in sect for _ in d['correct']])


# In[ ]:

get_ipython().magic('time gmod = Word2Vec(brown.sents(), **update(gparams))')
ev(gmod)


# In[ ]:

from sklearn.preprocessing import LabelEncoder

def to_ints(wds):
    le = LabelEncoder().fit(wds)
    ints = le.transform(wds)
    return list(ints), le

def to_list_gen(f):
    @wraps(f)
    def f2(*a, **kw):
        gen = f(*a, **kw)
        return (list(x) for x in gen)
    return f2

def tokenize(wds):
    alpha_re = re.compile(r'[A-Za-z]')
    return [w for w in wds if alpha_re.search(w)]

def prune_words(wds, keep_n_words=30000, min_counts=None):
    if (keep_n_words is None) and (min_counts is None):
        return wds
    cts = Counter(wds)
    if min_counts is not None:
        return [w for w in wds if cts[w] >= min_counts]
    elif keep_n_words is not None:
        keeps = set(sorted(cts, key=cts.get, reverse=True)[:keep_n_words])
        
    return [w for w in wds if w in keeps]

# c2 = prune_words(brown.words(), keep_n_words=10)


# In[ ]:

toks, le = to_ints(nopunct)


# In[ ]:

def from_gensim_params(params, cnf, **upkw):
    gsim2param_names = dict(negative='K', alpha='eta', size='N')
    newparams = {pn: params[gn] for gn, pn in gsim2param_names.items()}
    newparams['C'] = params['window'] * 2
    cnf2 = update(cnf, **newparams, **upkw)
    return cnf2

cnff = Conf(from_gensim_params(gparams, cnfe, dir='cache/v13'))
# cnff


# In[ ]:

params = gparams
params


# In[ ]:

class word2vec(object):
    def __init__(self, words, cnf, neg_sampler=to_list_gen(neg_sampler_np),
                 keep_n_words=None, min_counts=None, **sgd_kwds):
        text = prune_words(tokenize(words), keep_n_words=keep_n_words, min_counts=min_counts)
        
        self.text = text
        self.toks, self.le = to_ints(self.text)
        self.cnf = update(cnf, term={})
        self.W = wut.init_w(len(self.le.classes_), cnf.N)
        self.sgd_kwds = sgd_kwds
        self.neg_sampler = neg_sampler(self.toks, cnf.K)
        
    def run(self, term=None, **sgd_kwds):
        cnf = self.cnf
        if term:
            cnf = update(cnf, term=term)
        res = sgd(W=self.W.copy(), corp=self.toks, neg_sampler=self.neg_sampler,
                  cf=cnf, vc=self.le.classes_, **z.merge(self.sgd_kwds, sgd_kwds))
        self.W, self.cnf = res
        
    @property
    def df(self):
        return DataFrame(self.W.copy(), index=self.le.classes_)

modf = word2vec(brown.words(), cnff, neg_sampler=neg_sampler_j, keep_n_words=None, min_counts=5)


# In[ ]:

with open('cache/txt.txt','w') as f:
    f.write('\n'.join(modf.text))


# In[ ]:

fast_opts = dict(ns_grad=ns_grad_jit, sliding_window=sliding_window_jit)


# In[ ]:

get_ipython().run_cell_magic('time', '', "for i in range(4):   \n    modf.run(term={}, **fast_opts)  # 'iters': 10000\n!say done")


# In[ ]:

modf.run(term={}, **fast_opts)


# In[ ]:

gmod = 


# In[ ]:

gparams


# In[ ]:

Word2Vec


# In[ ]:

modf.cnf


# ## Evaluate

# In[ ]:




# In[ ]:



def partition(pred, iterable):
    'Use a predicate to partition entries into false entries and true entries'
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = it.tee(iterable)
    return it.filterfalse(pred, t1), ifilter(pred, t2)

with open('src/questions-words.txt', 'r') as f:
    qlns = f.read().splitlines()


# In[ ]:

with open('src/questions-words.txt', 'r') as f:
    qlns = f.read().splitlines()


# modf.df = DataFrame(modf.W.copy(), index=modf.le.classes_)
# W2 = modf.df

# In[ ]:

del W2


# In[ ]:

sections, qs = partition(lambda s: not s.startswith(':'), qlns)
qs = list(qs)
# allwds = set(modf.df.index)
# del allwds


# In[ ]:




# In[ ]:

cts = Series(Counter(modf.text)) # / len(modf.text)
assert (cts.index == modf.le.classes_).all()
keeps = cts >= 5
Wk = modf.df.divide(norm(modf.df, axis=1), axis=0)
Wsmall = Wk[keeps]
assert (norm(Wk, axis=1).round(4) == 1).all(), 'Not normalized'
# assert (Wk.sum(axis=1).round(4) == 1).all(), 'Not normalized'


# In[ ]:

get_ipython().magic('lprun -s -f eval eval_qs(Wk, lim=500)')


# In[ ]:

def to_vec(w, W):
    if isinstance(w, str):
        return W.ix[w]
    return w

def to_vec2(w, W, wd2row=None):
    if isinstance(w, str):
        return W.values[wd2row[w]]
    return w

neg = lambda x: -x

def combine(plus=[], minus=[], W=None, wd2row=None):
    to_vec_ = partial(to_vec, W=W)
    vecs = map(to_vec_, plus) + map(z.comp(neg, to_vec_), minus)
    v = sum(vecs) / len(vecs)
    return v / norm(v)

def combine2(plus=[], minus=[], W=None, wd2row=None):
    # plus_ix = [wd2row[p] for p in plus]
    # minus_ix = [wd2row[p] for p in minus]
    ixs = [wd2row[p] for p in plus + minus]
    vecs1 = Wk.values[ixs]
    to_vec_ = partial(to_vec2, W=W, wd2row=wd2row)
    
    vecs = map(to_vec_, plus) + map(z.comp(neg, to_vec_), minus)
    vecs = np.array(vecs)
    return combine_(vecs)
    v = sum(vecs) / len(vecs)
    return v / norm(v)


@nopython
def combine_(vecs):
#     v = sum(vecs) / len(vecs)
    v = np.sum(vecs, axis=0) / len(vecs)
    return v / np.linalg.norm(v)


# In[ ]:

aa = np.array(vecs)
aa.shape


# In[ ]:

combine_(aa)


# In[ ]:

get_ipython().magic('lprun -s -f combine2 combine2(plus=[b, c], minus=[a], W=Wk, wd2row=wd2ix)')


# In[ ]:

wvec = combine(plus=[b, c], minus=[a], W=Wk)
wvec[:5]


# In[ ]:

mk_wd2row = lambda W: dict(zip(W.index, count()))


# In[ ]:

wd2ix = mk_wd2row(Wk)
wvec2 = 
wvec2[:5]


# In[ ]:

get_ipython().magic('timeit combine(plus=[b, c], minus=[a], W=Wk)')
get_ipython().magic('timeit combine2(plus=[b, c], minus=[a], W=Wk, wd2row=wd2ix)')


# In[ ]:

get_ipython().magic('timeit np.array(vecs)')


# In[ ]:

combine2(plus=[b, c], minus=[a], W=Wk, wd2row=wd2ix)


# In[ ]:

(wvec == wvec2).all()


# In[ ]:

@nopython
def index(wd2ix):
    
    return wd2ix


# In[ ]:

wds, arr, 


# In[ ]:

index({'abc': 123})


# Wv = Wk.values
# f1 = lambda: Wk.ix['youngster']
# f2 = lambda: Wk.iloc[-18]
# f3 = lambda: Wv[-18]
# assert np.allclose(f1(), f3())
# assert np.allclose(f1(), f2())
# 
# %timeit f1()
# %timeit f2()
# %timeit f3()

# In[ ]:




# In[ ]:

def eval(q, Wall=None, Wsmall=None, Wnorm=None, allwds=None):
    qwds = a, b, c, d = q.split()
    if allwds is None:
        print('Warning: precalculate `allwds`')
        allwds = set(Wall.index)
    missing_wds = {a, b, c} - allwds
    if missing_wds:
        # print(u'\u2639', end='')
#         print(u'.', end='')
        return False
    wvec = combine(plus=[b, c], minus=[a], W=Wall)
    [closest] = wut.get_closestn(wd=wvec, W=Wsmall, n=1, exclude=[a, b, c], just_word=1)
    ans = closest == d
        
    return ans

        
def eval_qs(Wsmall, Wall, lim=None):
    Wn = np.linalg.norm(Wsmall, axis=1)
    allwds = set(Wall.index)
    sm = 0
    for ql in qs[:lim]:
        print(list(allwds)[:5])
        e = eval(ql, Wall=Wall, Wsmall=Wsmall, Wnorm=Wn, allwds=allwds)
        if e:
#             print('!', end='')
            print(ql, end=':')
            sm += 1
    return sm
    #     break


# In[ ]:

get_ipython().magic('pinfo2 Wk.ix')


# In[ ]:

Wk._


# In[ ]:

# %lprun -s -f eval eval_qs(Wk, lim=500)
get_ipython().magic('lprun -s -f combine eval(ql, Wall=Wk, Wsmall=Wk, Wnorm=norm(Wk, axis=1), allwds=set(Wk.index))')


# In[ ]:

# %lprun eval(ql, Wall=Wk, Wsmall=Wk, Wnorm=norm(Wk, axis=1))


# In[ ]:

sc


# In[ ]:

sc = eval_qs(Wk, Wk, lim=50)


# In[ ]:

sc


# In[ ]:

Wk.shape


# In[ ]:

gmod.syn0norm.shape


# In[ ]:




# In[ ]:




# In[ ]:

a, b, c, d = 'boy', 'girl', 'brothers', 'sisters'


# In[ ]:

norm = np.linalg.norm
reload(wut);


# In[ ]:

closests = wut.get_closestn.closests
print(len(closests))
closests


# In[ ]:

norm(Wk, axis=1)


# In[ ]:

wvec = combine(plus=[b, c], minus=[a], W=Wk)
cl = wut.get_closestn(wd=wvec, W=Wk, Wnorm=None, n=1, exclude=[a, b, c], just_word=1)
cl


# In[ ]:

wut.cdist(wvec, Wk)[Wk.index.isin([a, b, c])]


# In[ ]:




# In[ ]:

sis = Wsmall.ix['sisters']
(sis @ wvec) / (norm(sis) * norm(wvec))


# In[ ]:

ga, gb, gc, gd = gmod.vocab[a], gmod.vocab[b], gmod.vocab[c], gmod.vocab[d],
ga, gb, gc, gd = gmod.syn0norm[ga.index], gmod.syn0norm[gb.index], gmod.syn0norm[gc.index], gmod.syn0norm[gd.index]
gv = (gb + gc - ga) / 3
gv /= norm(gv)


# In[ ]:

gv


# In[ ]:

norms = gmod.syn0norm @ gv
isort = np.argsort(norms)[::-1][:20]


# In[ ]:

[gmod.index2word[i] for i in isort]


# In[ ]:

norms[isort]


# In[ ]:

isort


# In[ ]:

gmod.syn0norm.shape


# In[ ]:




# In[ ]:

get_ipython().magic('pinfo get_closests')
get_closests 


# In[ ]:

sc


# In[ ]:

len(Wk)


# In[ ]:

gsc = gmod.accuracy('src/questions-words.txt')


# In[ ]:




# In[ ]:

gsc_dct = {doc.pop('section'): doc for doc in gsc}
gfound = {w for set_ in gsc_dct['total']['correct'] for w in set_[:3]}


# In[ ]:




# In[ ]:

b, c, gmod.most_similar(positive=[b])


# In[ ]:

gmod.most_similar(positive=[b, c], negative=[a])


# In[ ]:




# In[ ]:

gsc_dct['family']['correct']


# In[ ]:

for doc in gsc:
    print(doc['section'], end=': ')
    print(len(doc['correct']))
#     break


# In[ ]:

list(gsc)


# In[ ]:




# In[ ]:

cnfs = update(cnfe, dir='cache/slow')
mod = word2vec(brown.words(), cnfs)


# In[ ]:

ls src/


# In[ ]:

get_ipython().magic("time mod.run(term={})  # 'iters': 10000")


# ## Gensim benchmarking

# In[ ]:

import gensim
from gensim.models.word2vec import Word2Vec


# In[ ]:

get_ipython().magic('pinfo Word2Vec')


# In[ ]:

gparams


# In[ ]:

gensim_sents = brown.sents() # [s.orth_.split() for s in atks.sents]

param_vals = dict(
    sample=[1e-3, 5e-3, 1e-2, 5e-2, .1, .15, 0],
    negative=range(0, 8),  #[5, 7, 10, 12, 15, 17],
    window=range(2, 20),
    sg=[0, 1],
    size=np.arange(1, 40) * 4,
    alpha=[0.05, 0.025, 0.01, 0.005],
    iter=range(1, 5),
    
)

param_lst = ['alpha', 'iter', 'negative', 'sample', 'sg', 'size', 'window']
assert sorted(param_vals) == param_lst, 'Need to update param_lst'

cs = param_lst + ['score']
to_param_dct = lambda xs: OrderedDict(zip(param_lst, xs))
from_param_dct = lambda dct, cs=param_lst: [dct[c] for c in cs]

# param_vals


# get_closests(to_param_dct([0.025, 4.0, 0.0, 0.0, 1.0, 80.0, 4.0]))

# In[ ]:

to_param_dct([0.025, 4.0, 0.0, 0.0, 1.0, 80.0, 4.0])


# In[ ]:

def get_closest(k, v, poss=param_vals):
    return min(poss[k], key=lambda x: abs(v - x))

def get_closests(dct, poss=param_vals):
    return {k: get_closest(k, v, poss=poss) for k, v in dct.items()}

def param_gen(params, n_iters=None):
    counter = count() if n_iters is None else range(n_iters)
    for i in counter:
        yield {k: nr.choice(v) for k, v in param_vals.items()}

def run_model(n=None, perfs=None, lfname='cache/log.csv'):
    pg = param_gen(param_vals, n)

    for params in pg:
        st = time.time()
        model = Word2Vec(sentences=gensim_sents, workers=4, **params)
        perf = dict(params)
        perf['score'] = score(model)
        perfs.append(perf)
        print('time: {:.2f}'.format(time.time() - st))
      
        if not os.path.exists(lfname):
            mode, header = 'w', True
        else:
            mode, header = 'a', False
        with open(lfname, mode) as f:
            DataFrame([perf])[cs].to_csv(f, header=header, sep='\t')
        sys.stdout.flush()
    return perfs

def score(model, restrict_vocab=10000):
    acc = model.accuracy('src/questions-words.txt', restrict_vocab=restrict_vocab)
    [tot] = [d for d in acc if d['section'] == 'total']
    print(len(tot['correct']), '/', len(tot['incorrect']) + len(tot['correct']), end=' ')
    return len(tot['correct'])


# In[ ]:

param_ex = next(param_gen(param_vals, None))
ex_vals = map(itg(1), sorted(param_ex.items()))
to_param_dct(ex_vals)


# In[ ]:

para


# In[ ]:

res = minimize(rosen, x0, method='nelder-mead',
               options={'xtol': 1e-8, 'disp': True})


# In[ ]:

def eval_func(paramlist):
    print(paramlist)
    param_dct_ = to_param_dct(paramlist)
    param_dct = {k: get_closest(k, v) for k, v in param_dct_.items()}
    print(param_dct)
    model = Word2Vec(sentences=gensim_sents, workers=4, **param_dct)
    return score(model)


# In[ ]:




# In[ ]:

from scipy.optimize import minimize


# In[ ]:

get_ipython().magic('pinfo minimize')


# In[ ]:

# perfs = []
perfs = run_model(n=None, perfs=perfs)
## End Gensim benchmarking


# In[ ]:

cs


# In[ ]:

perfs[:4][cs]


# In[ ]:

Series(list(cts.values())).value_counts(normalize=1)


# In[ ]:

modf.W.shape


# In[ ]:




# In[ ]:

z.merge({2: 4}, {2: 3})


# In[ ]:

def f(**kw):
    print(kw)
    
f(a=2, **{'a': 3})


# In[ ]:

# neg_sampler=ngsamp, 


# In[ ]:

ngsamp = neg_sampler_j(toki, cnfe.K)


# In[ ]:


get_ipython().magic("lprun -T lp5.txt -s -f sgd sgd(W=We.copy(), corp=toki, cf=update(cnfe, term={'iters': 10000}), **fast_opts) # ls[:20]")


# In[ ]:




# In[ ]:

cnfe.


# In[ ]:

gmod


# In[ ]:

gmod.most_similar('politician')


# In[ ]:

brown.sents()


# In[ ]:

get_ipython().magic('time sgd(W=We.copy(), corp=toki, cf=update(cnfe, term=dict()))')


# In[ ]:

get_ipython().magic('time sgd(W=We.copy(), corp=toki, cf=update(cnfe, term=dict()), **fast_opts)')


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

z

