
# coding: utf-8

# ## Word2vec
# A difficulty of working with text data is that using each word of a large vocabulary as a feature requires working with large dimensions. While sparse matrix data structures make manipulating these vectors tractable, there are whole classes of algorithms that do not work well or at all on sparse high dimensional data. 
# 
# A recent development called word2vec allows for words to be represented as dense vectors of much smaller dimensions (on the order of $\mathbb{R}^{100}$). This algorithm yields representations of words such that words appearing in similar contexts will lie close to one another in this low dimensional vector space. Another interesting feature of this algorithm is that the representation does a good job at inferring analogies. [This gensim demo](http://rare-technologies.com/word2vec-tutorial/#app) of the algorithm shows word2vec inferring that *dogs* is to *puppies* as *cats* is to *kittens*:

# In[ ]:

from IPython.display import Image
Image(filename='kittens.png')


# 
# At a high level, the skip-gram flavor of this algorithm looks at a word and its surrounding  words, and tries to maximize the probability that the word's vector representation predicts those actual words occurring around it. If it is trained on the phrase *the quick brown fox jumps*, the word2vec input representation of the word *brown* would yield a high dot product with the output vectors for the words *the, quick, fox* and *jumps*. And if the algorithm is trained on a lot more text, there's a good chance that it will start to learn that the words $brown$ and $red$ appear in similar contexts, so that their vector representations will be pretty close to one another's.
# 
# ## Speed
# I'm constantly trying to navigate the trade-off from simultaneously maximizing my laziness and the speed of my code. Aiming for the lazy side, I originally wanted to experiment with using [autograd](https://github.com/HIPS/autograd) to write my gradient updates for me, and as nice as it was to not have to write out the gradient calculation, this pretty quickly bubbled to the top as one of the major bottlenecks, leaving me to write the gradients manually by faster means.
# 
# While numpy gives a big speed boost over plain python, cython (used by the go-to word2vec implementation, [gensim](https://github.com/piskvorky/gensim)), gives a major performance improvement beyond numpy, with speeds often comparable to code written in C. Part of the cost of this speed up is that writing in Cython, while more pythonic than C, seems to require additional type annotations and syntactic elements, making it less readable (at least for me). My goal here has been to make word2vec as close to gensim's cython implementation as possible while sticking to Python, so I settled on Numba, a numeric JIT compiler that uses LLVM and supports a subset of Python and numpy.
# 
# While I ran into some limitations of numba, rewriting the inner loops in Numba functions ended up giving a significant speed-boost. I did a lot of profiling and iterating to find the major bottlenecks of the gradient descent learning, and I've left the original numpy and improved numba versions of some of these functions for comparison.

# # Let's get started

# In[ ]:

get_ipython().run_cell_magic('javascript', '', "var csc = IPython.keyboard_manager.command_shortcuts\ncsc.add_shortcut('Ctrl-k','ipython.move-selected-cell-up')\ncsc.add_shortcut('Ctrl-j','ipython.move-selected-cell-down')\ncsc.add_shortcut('Shift-m','ipython.merge-selected-cell-with-cell-after')")


# In[ ]:

from project_imports import *
get_ipython().magic('matplotlib inline')


# In[ ]:

import wordvec_utils as wut; reload(wut);
from wordvec_utils import Cat, WordVectorizer, update, Conf
import utils as ut; reload(ut);
from utils import take, ilen

from numba_utils import ns_grad as ns_grad_jit
import numba_utils as nbu; reload(nbu)
from autograd import numpy as npa, grad

from numba import jit
nopython = jit(nopython=True)


# ## Objective functions
# 
# The standard skip-gram objective function comes from taking the softmax probability of each of the actual context word vectors dotted with the input ($u_{c,j^*_c}$) and multiplying them together:
# 
# \begin{align}
# E & = -\log \prod_{c=1} ^{C}
#     \frac {\exp (u_{c,j^*_c})}
#           {\sum_{j'=1} ^ V \exp(u_{j'})} \\
#   & = -\sum^C_{c=1} u_{j^*_c} + C \cdot \log \sum ^ V _{j'=1} \exp(u_j')
# \end{align}
# 
# See [word2vec Parameter Learning Explained](http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf) for a detailed explanation, and note that the algorithm keeps 2 representations for each word, an input vector from $W$, and an output vector from $W'$ (`W1` and `W2` in the code).
# Though I ended up using a different objective called negative sampling, I kept this original objective below as `skipgram_likelihood`, written as a nested function since autograd requires single-argument functions to differentiate:

# In[ ]:

def skipgram_likelihood(wi, cwds, dv=None):
    wix = dv.get(wi)
    cixs = dv.get(cwds)
    C = len(cwds)

    def logloss(Wall):
        W1, W2 = Cat.split(Wall)
        h = W1[wix, :]  # ∈ ℝⁿ
        u = np.dot(h, W2)
        ucs = u[cixs]
        return -np.sum(ucs) + C * np.log(np.sum(np.exp(u)))
    return logloss


# The return value of `logloss` should pretty clearly resemble the equation above.
# 
# # Negative sampling
# ## Gradient
# 
# After reading a bit more about word2vec, I found out about an extension to the skip-gram model called negative sampling that efficiently generates better word vectors. The basic idea is that in addition to training a word vector with $C$ words that *do* appear around it, the vector should also be trained with $K$ words randomly chosen from the rest of the text, as negative examples of what the vector should *not* predict in its context. 
# 
# As a side-note to keep up with the notation taken from [word2vec Parameter Learning Explained](http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf), the term $\boldsymbol v_{w_O}'$ refers to a word vector ($\boldsymbol v$) that is from the output vector matrix ($\boldsymbol v'$) representing an output word $w_O$. Notation aside, the gradient for the negative sampling extension is relatively straightforward. For each true context word vector $\boldsymbol v_{w_O}'$ appearing close to the input word vector $h$, we'll draw $K$ word vectors $\boldsymbol v_{w_i}$ at random from the corpus. The objective is computed by adding the log of the sigmoid of the word vector dotted with either the true or false context word (the negative samples are negated): 

# 
# $$
# E = -\log \sigma(\boldsymbol v_{w_O}^{\prime T} \boldsymbol h)
#     - \sum^K _{i=1} \log \sigma (-\boldsymbol v_{w_i} ^{\prime T} \boldsymbol h)
# $$
# 
# The gradient with respect to $\boldsymbol v_{w_j}^{\prime T} \boldsymbol h$ is then as follows, where $t_j$ is an indicator for whether $w_j$ actually appears in the context:
# 
# $$
#     \frac{\partial E}
#          {\partial \boldsymbol v_{w_j}^{\prime T} \boldsymbol h}
#          = \sigma(\boldsymbol v_{w_j}^{\prime T} \boldsymbol h) -t_j.
# $$
# 
# The numpy gradient is written below as `ns_grad`; the numba gradient `ns_grad_jit` is similar, but all of the functions used in a JIT'd function must also be numba-JIT'd, so I moved them all out to the `numba_utils` module. 
# 
# The schema I adopted to calculate gradients takes a subset of the matrix formed by concatenating the input and output parameter matrices together ($W || W'$). If the vector dimension $N$ is 100, and there are 1000 words in the vocabulary, this matrix will be in $\mathbb{R}^{1000 \times 200}$. The gradient function takes a subset of this matrix with $K + 2$ rows. The first row represents the input word, the next represents the true context word, and the rest represent the $K$ negative samples.
# 
# Note that since the first vector is from the input parameter matrix $W$, and these functions are operating on the input and output matrices concatenated, we only care about the first $N$ entries of the first row. Similarly, we only care about the last $N$ entries for the rest of the output vectors. This schema should be more clear from the `get_vecs1` function below that extracts the vectors from the relevant subset of the concatenated parameter matrices. This also means that for the gradients of the concatenated matrix, half of the entries in each row will be zero.

# In[ ]:

getNall = lambda W: W.shape[1] // 2
gen_labels = lambda negsamps: [1] + [0] * len(negsamps)
sig = lambda x: 1 / (1 + np.exp(-x))


def get_vecs1(Wsub):
    N = getNall(Wsub)
    h = Wsub[0, :N]  # ∈ ℝⁿ
    vwo = Wsub[1, N:]
    negsamps = Wsub[2:, N:]
    return h, vwo, negsamps

def ns_loss_grads(h: 'v[n]', vout: '[v[n]]', label: 'v[n]'):
    dot = sig(vout @ h) - label
    return dot * vout, dot * h

def ns_grad(Wsub):
    h, vwo, negsamps = get_vecs1(Wsub)
    N = getNall(Wsub)
    Wsub_grad = np.zeros(Wsub.shape)

    for i, vout, label in izip(count(1), it.chain([vwo], negsamps), gen_labels(negsamps)):
        hgrad, vgrad = ns_loss_grads(h, vout, label)
        Wsub_grad[0, :N] += hgrad
        Wsub_grad[i, N:] += vgrad
    return Wsub_grad


# ## Gradient check
# The following gradient checking functionality based on [the UFLDL tutorial](http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/) uses simple calculus to ensure that the gradients are working as expected.

# In[ ]:

def ns_loss(h, vwo, vwi_negs):
    """This should be called on the subset of the matrix (Win || Wout')
    determined by row indices `wi, win_ix, negwds`.
    """
    negsum = 0
    for j in xrange(len(vwi_negs)):
        negsum += np.log(sig(-vwi_negs[j] @ h))
        
    return -np.log(sig(vwo @ h)) - negsum


def ns_loss_vec(h, vwo, vwi_negs):
    """This should be called on the subset of the matrix (Win || Wout')
    determined by row indices `wi, win_ix, negwds`.
    """
    return -np.log(sig(vwo @ h)) - np.sum(np.log(sig(-vwi_negs @ h )))


# In[ ]:

def J(Wsub, loss=ns_loss):
    N = getNall(Wsub)
    h, vwo, vwi_negs = get_vecs1(Wsub)
    return loss(h, vwo, vwi_negs)

def check_grad_(W, i: int=None, j: int=None, eps=1e-6, J: 'Callable'=None):
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


# In[ ]:

def siga(x):
    return 1 / (1 + npa.exp(-x))

def mk_ns_loss_a(N):
    """Return a loss function that works on an N-dimensional
    representation. This takes a single argument, the subset of
    the input and output matrices that correspond to the input
    word (first row), true contest word (second row) and K
    negative samples (rest of the rows). Since it takes a
    single argument, the gradient can automatically be
    calculated by autograd"
    """
    def ns_loss_a(Wsub):
        h = Wsub[0, :N]
        vwo = Wsub[1, N:]
        vwi_negs = Wsub[2:, N:]
        vwo_h = npa.dot(vwo, h)
        vwi_negs_h = npa.dot(vwi_negs, h)
        return  -npa.log(siga(vwo_h)) - npa.sum(npa.log(siga(-vwi_negs_h)))
    return ns_loss_a

mk_ns_grad_a = z.comp(grad, mk_ns_loss_a)


# In[ ]:

N_ = 50; W = wut.init_w(1000, N_, seed=1); Wsub = W[:8]


# In[ ]:

np_check = lambda x: approx_grad(x, partial(J, loss=ns_loss))
np_vec_check = lambda x: approx_grad(x, partial(J, loss=ns_loss_vec))
ns_grad_auto = mk_ns_grad_a(N_)


# In[ ]:

def grad_close(f, grd=ns_grad(Wsub)):
    """Check that a given gradient function result agrees
    with ns_grad. Print out norm of the difference."""
    grd2 = f(Wsub)
    close = np.allclose(grd, grd2)
    close_ = '√' if close else 'x'
    print('{} Diff: {}'.format(close_, np.linalg.norm(grd - grd2)))
    return close


# In[ ]:

assert grad_close(np_check)
assert grad_close(np_vec_check)
assert grad_close(ns_grad_auto)
assert grad_close(ns_grad)
assert grad_close(ns_grad_jit)


# In[ ]:

get_ipython().magic('timeit np_check(Wsub)      #  48.9 ms per loop')
get_ipython().magic('timeit np_vec_check(Wsub)  #  33.8 ms per loop')
get_ipython().magic('timeit ns_grad_auto(Wsub)  # 856 µs per loop')
get_ipython().magic('timeit ns_grad(Wsub)       #  53.8 µs per loop')
get_ipython().magic('timeit ns_grad_jit(Wsub)   #   6.14 µs per loop')


# The first two gradient checking versions are extremely slow as expected from the naive and unpythonic style. The numpy implementation gives a huge improvement over these, and numba gives us another order of magnitude speedup over this. The autograd version looks like it falls at about the logarithmic midpoint between the naive checking and handwritten implementations.
# 
# ### Draw negative samples
# 
# As a foreshadowing of performance bottlenecks that my original implementation ran into, I have a few versions of a function that chooses words from the text at random, that increase in performance. They all draw words randomly according to the unigram$^{3/4}$ distribution, which is similar to the unigram distribution, but boosts the probability of less frequent items:

# In[ ]:

uni = np.linspace(0, 1, 100)
uni34 = uni ** (3/4)
plt.plot(uni, uni34 - uni);


# Most of the samplers were written as generators, except for the last one, `neg_sampler_inplace`, which modifies an array in-place with random samples. As much as this in-place modification conflicts with my usual approach, it ended up being necessary for a final numba performance boost.
# 
# The generator sampler I ended up going with, `neg_sampler_jit_pad`, is padded with a couple of dummy entries. I originally drew some negative samples `negsamps` from one of these generators, and then created the array to index $W$ by prepending `negsamps` with `w` and `c`, copying these to a new array. Usually this is ok, but I found that copying `negsamps` to a new array in the inner loop caused a noticeable delay, so I wrote `neg_sampler_jit_pad` to return 2 extra empty elements at the beginning to be able to insert `w` and `c` inplace without copying to a new array.

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

def cum_prob_uni(xs, pow=.75):
    ug = unigram(xs, pow=pow)
    return ug.Prob.cumsum().div(ug.Prob.sum()).values


def neg_sampler_pd(xs, K, pow=.75):
    "Simplest, but slowest sampler using built-in pandas sample function"
    ug = unigram(xs, pow=pow)
    for seed in count():
        yield ug.Word.sample(n=K, weights=ug.Prob, random_state=seed, replace=True)
        
        
def neg_sampler_np(xs, K, cache_len=1000, use_seed=False, pow=.75, ret_type=np.ndarray):
    "Faster neg. sampler without the pandas overhead"
    ug = unigram(xs, pow=pow)
    p = ug.Prob.values / ug.Prob.sum()
    a = ug.Word.values

    def neg_sampler_np_():
        for seed in count():
            if use_seed:
                nr.seed(seed)
            Wds = nr.choice(a, size=(cache_len, K), p=p)
            for wds in Wds:
                yield wds
                
    agen = neg_sampler_np_()            
    if ret_type == list:
        return imap(list, agen)
    assert ret_type == np.ndarray, 'Only defined for type in {np.ndarray, list}'
    return agen


def neg_sampler_jit(xs, K, pow=.75, ret_type=list):
    cum_prob = cum_prob_uni(xs, pow=pow)
    sampler = {
        list:     nbu.neg_sampler_jitl_,
        np.ndarray: nbu.neg_sampler_jita_,
    }[ret_type]
    return sampler(cum_prob, K)

def neg_sampler_jit_pad(xs, K, pow=.75, ret_type=list, pad=0):
    cum_prob = cum_prob_uni(xs, pow=pow)
    sampler = {
        list:     nbu.neg_sampler_jitl_pad,
        np.ndarray: nbu.neg_sampler_jita_pad,
    }[ret_type]
    return sampler(cum_prob, K, pad=pad)

def neg_sampler_inplace(xs, K, pow=.75, pad=0, ret_type=np.ndarray):
    cum_prob = cum_prob_uni(xs, pow=pow)
    a = np.empty(K + pad, dtype=np.int64)
    
    @nopython
    def neg_sampler_jit_pad_arr_(a):
        for i in xrange(pad, K + pad):
            a[i] = nbu.bisect_left_jit(cum_prob, nr.rand())
    return a, neg_sampler_jit_pad_arr_

def neg_sampler_inplace_gen(xs, K, pow=.75, pad=0, ret_type=np.ndarray):
    a, neg_sampler_jit_pad_arr_ = neg_sampler_inplace(xs, K, pow=pow, pad=pad)
    for _ in it.repeat(None):
        neg_sampler_jit_pad_arr_(a)
        yield a[pad:].copy()
#         for i in a[pad:]:
#             yield i


# ### Check distributions
# 
# Just as a sanity check that the different implementations do the same thing, I randomly generate words according to how frequently they occur in the text with each of the samplers and scatter-plot them against the actual word frequency to check that they mostly lie on $y=x$. 

# In[ ]:

from sklearn.preprocessing import LabelEncoder
from nltk.corpus import brown
some_text = take(brown.words(), int(1e6))


# In[ ]:

le = LabelEncoder()
smtok = le.fit_transform(some_text)


# In[ ]:

class NegSampler:
    """Container for the sampler generator functions.
    This keeps track of the number of samples $K$ and the padding."""
    def __init__(self, sampler, toks, K=None, ret_type=None, pad=None, nxt=True, **kw):
        if pad is not None:
            kw['pad'] = pad
        if ret_type is not None:
            kw['ret_type'] = ret_type
        
        self.sampler = sampler(toks, K, **kw)
        if nxt:
            a = next(self.sampler)
            if ret_type:
                assert isinstance(a, ret_type)
        self.toks = toks
        self.K = K
        self.ret_type = ret_type
        self.pad = pad or 0
        
    def __iter__(self):
        return self.sampler
    
    def __next__(self):
        return next(self.sampler)
    
    def __repr__(self):
        return ('NegSampler(pad={pad}, K={K}, type={ret_type.__name__})'
                .format(pad=self.pad, K=self.K, ret_type=self.ret_type))


# In[ ]:

gen_pd =       NegSampler(neg_sampler_pd, smtok, 8)
gen_npl =      NegSampler(neg_sampler_np, smtok, 8, ret_type=list)
gen_npa =      NegSampler(neg_sampler_np, smtok, 8, ret_type=np.ndarray)
gen_jitl =     NegSampler(neg_sampler_jit, smtok, 8, ret_type=list)
gen_jitl_pad = NegSampler(neg_sampler_jit_pad, smtok, 8, ret_type=list, pad=2)
gen_jita =     NegSampler(neg_sampler_jit, smtok, 8, ret_type=np.ndarray)
gen_jita_pad = NegSampler(neg_sampler_jit_pad, smtok, 8, ret_type=np.ndarray, pad=2)
# gen_jit_pada = NegSampler(neg_sampler_jit_pad_arr, smtok, 8, C=5, pad=2)


# In[ ]:

gen_jita_padi = NegSampler(neg_sampler_inplace_gen, smtok, 8, ret_type=np.ndarray, pad=2)


# In[ ]:

# a_, inplace = neg_sampler_inplace(smtok, 8, pow=.75, pad=2)
# run_n2d = lambda gen, n=10000, w=5: ilen(x for xs in it.islice(gen, n // w) for x in xs)
# run_part = lambda gen, n=10000, w=5: ilen(x for xs in it.islice(z.partition(w, gen), n // w) for x in xs)


# In[ ]:

run_n = lambda gen, n=10000: ilen(it.islice(gen, n))
get_ipython().magic('timeit run_n(gen_pd, n=1000)')
get_ipython().magic('timeit run_n(gen_npl)')
get_ipython().magic('timeit run_n(gen_npa)')
get_ipython().magic('timeit run_n(gen_jitl)')
get_ipython().magic('timeit run_n(gen_jitl_pad)')
get_ipython().magic('timeit run_n(gen_jita)')
get_ipython().magic('timeit run_n(gen_jita_pad)')
get_ipython().magic('timeit run_n(gen_jita_padi)')
# %timeit run_n2d(gen_jit_pada, w=5)


# In[ ]:

n = 100000
csp = Series(Counter(x for xs in it.islice(gen_pd, n // 100) for x in xs))
csnp = Series(Counter(x for xs in it.islice(gen_npl, n) for x in xs))
csj = Series(Counter(x for xs in it.islice(gen_jita_pad, n) for x in xs[2:]))
cs_ip = Series(Counter(x for xs in it.islice(gen_jita_padi, n) for x in xs))
# cs_ip = Series(Counter(x for _ in xrange(n) for x in inplace(a_) or a_[2:]))


ug = unigram(smtok, pow=.75)
cts = DataFrame({'Numba': csj, 'Numpy': csnp, 'Pandas': csp, 'Inplace': cs_ip}).fillna(0)
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
plot_dist(xcol='Numba', subplt=141)
plot_dist(xcol='Numpy', subplt=142)
plot_dist(xcol='Pandas', subplt=143)
plot_dist(xcol='Inplace', subplt=144)


# As these plots show, the samplers seem to be drawing words according to the expected distribution. The pandas sampler was so slow that I had to reduce the number of draws by a couple orders of magnitude, so the plot is a lot noisier than the others, but still roughly on track. The in-place sampler appears to be slower on the microbenchmarks, but gives an improvement when restructuring the gradient update routine.
# 
# ### Sliding window
# It turned out another significant bottleneck of the gradient descent routine was the sliding window code that iterates over the entire corpus, yielding a word and its context words at each point. 
# 
# For the first few words in the text sequence, there are less than $C$ surrounding context words, so some checking is required. For the numba function, I used a specialized iterator at the beginning and end to avoid checking the max and min indices at every step.
# Just like in the negative sampler section, I first wrote the sliding window functions in a generator style, and converted the fastest version to an inplace modification style.

# In[ ]:

def sliding_window(xs, C=4, start_pos=0):
    """Iterate through corpus, yielding input word
    and surrounding context words"""
    winsize = C // 2
    N = len(xs)
    for i, x in enumerate(xs, start_pos):
        ix1 = max(0, i-winsize)
        ix2 = min(N, i+winsize+1)
        yield x, xs[ix1:i] + xs[i + 1:ix2]

@nopython
def sliding_window_jit(xs, C=4):
    """Iterates through corpus, yielding input word
    and surrounding context words"""
    winsize = C // 2
    N = len(xs)
    for i in xrange(winsize):
        yield nbu.bounds_check_window(i, xs, winsize, N)
    for i in xrange(winsize, N-winsize):
        context = []
        for j in xrange(i-winsize, i+winsize+1):
            if j != i:
                context.append(xs[j])
        yield xs[i], context
    for i in xrange(N-winsize, N):
        yield nbu.bounds_check_window(i, xs, winsize, N)

@nopython
def sliding_window_jit_arr(xs, C=4):
    """Iterates through corpus, yielding input word
    and surrounding context words"""
    winsize = C // 2
    N = len(xs)
    for i in xrange(winsize):
        yield nbu.bounds_check_window_arr(i, xs, winsize, N)
    for i in xrange(winsize, N-winsize):
        context = np.empty(C, dtype=np.int64)
        for ci in xrange(winsize):
            context[ci] = xs[i - winsize + ci]
            context[winsize + ci] = xs[i + 1 + ci]
        yield xs[i], context
    for i in xrange(N-winsize, N):
        yield nbu.bounds_check_window_arr(i, xs, winsize, N)


# In[ ]:

@nopython
def sliding_window_inplace(xs, i, C=4):
    winsize = C // 2
    N = len(xs)
    if i < winsize:
        return nbu.bounds_check_window_arr(i, xs, winsize, N)
    elif i < N-winsize:
        context = np.empty(C, dtype=np.int64)
        for ci in xrange(winsize):
            context[ci] = xs[i - winsize + ci]
            context[winsize + ci] = xs[i + 1 + ci]
        return xs[i], context
    elif i < N:
        return nbu.bounds_check_window_arr(i, xs, winsize, N)
    raise ValueError('Out of bounds')
    
def sliding_window_ix_(xs, C=4):
    return (sliding_window_inplace(xs, i, C=C) for i in xrange(len(xs)))


# In[ ]:

samp_toks = nr.randint(0, 1e6, size=100005)
samp_toksl = list(samp_toks)
list(sliding_window_jit(samp_toksl[:10], C=4))
run_window = lambda f, toks=samp_toksl: [ut.ilen(xs) for xs in f(toks, C=4)]


# In[ ]:

to_lst = lambda xs: [(c, list(ys)) for c, ys in xs]
assert to_lst(sliding_window_ix_(samp_toks[:10], C=4)) == to_lst(sliding_window_jit_arr(samp_toks[:10], C=4))


# In[ ]:

get_ipython().magic('timeit run_window(sliding_window)')
get_ipython().magic('timeit run_window(sliding_window_jit)')
get_ipython().magic('timeit run_window(sliding_window_jit_arr, toks=samp_toks)')
get_ipython().magic('timeit run_window(sliding_window_ix_, toks=samp_toks)')


# ### Norm
# Since I'm using the concatenated input and output matrix, half of the `w, c, negsamps` submatrix entries are 0. A specialized norm function taking this structure into account gives about an order of magnitude speedup over the numpy one.

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


# ## Gradient descent
# That does it for optimizing most of the individual pieces.
# For the gradient descent routine, I'm passing all the hyper-parameters through a configuration dictionary, validated by [voluptuous](https://pypi.python.org/pypi/voluptuous). This allows specification of things like the context window size, learning rate, number of negative samples and size of the word vectors. Check the utility file `wordvec_utils.py` where it's defined for details on the meanings of the parameters.

# In[ ]:

all_text = list(brown.words())
dv = WordVectorizer().fit(all_text)
toki = [dv.vocabulary_[x] for x in all_text]
vc = dv.feature_names_


# In[ ]:

cnf = ut.AttrDict(
    eta=.1, min_eta=.0001, accumsec=0,
    N=100, C=4, K=6, iter=0, thresh=15, epoch=0,
    pad=0,
    term=dict(iters=None,
              secs=10
    ),
    dir='cache',
)
cnf = Conf(cnf)
cnf_ = cnf
del cnf
cfp = update(cnf_, pad=2)

W = wut.init_w(len(vc), cnf_.N, seed=1)
We = W.copy()


# ### Sgd
# One final speedup I got was from putting the entire inner loop routine into a numba JIT'd function, instead of calling each JIT'd function separately. Numba seems to reduce some of python's function call overhead, as well as the numpy indexing. For some reason, indexing into a numpy array (to calculate the gradient and then update the original matrix) still ended up being a significant bottleneck, though numba seemed to reduce it.

# In[ ]:

get_ipython().magic('load_ext line_profiler')


# In[ ]:

def mk_grad_update(jit=False, grad_func=ns_grad):
    if jit:
        grad_func = ns_grad_jit
        norm_func = grad_norm
        deco = nopython
    else:
        norm_func = np.linalg.norm
        deco = lambda x: x
        
    def grad_update(W, sub_ixs, eta):
        Wsub = W[sub_ixs]
        grad = grad_func(Wsub)
        gnorm = norm_func(grad)
        if gnorm > 5:  # clip gradient
            grad /= gnorm
        W[sub_ixs] = Wsub - eta * grad
        
    return deco(grad_update)

grad_update = mk_grad_update(jit=False)
grad_update_jit = mk_grad_update(jit=True)


#     def grad_update(W, sub_ixs, eta, ns_grad=ns_grad):
#         Wsub = W[sub_ixs]
#         grad = ns_grad(Wsub)
#         gnorm = np.linalg.norm(grad)  # grad_norm
#         if gnorm > 5:  # clip gradient
#             grad /= gnorm
#         W[sub_ixs] = Wsub - eta * grad
# 
#     @nopython
#     def grad_update_jit(W, sub_ixs, eta):
#         Wsub = W[sub_ixs]
#         grad = ns_grad_jit(Wsub)
#         gnorm = grad_norm(grad)
#         if gnorm > 5:  # clip gradient
#             grad /= gnorm
#         W[sub_ixs] = Wsub - eta * grad

# In[ ]:

@nopython
def grad_update_jit_pad(W, sub_ixs, eta, w=0, c=0):
    """If focus or context word are contained in negative samples,
    drop them before performing the grad update.
    """
    sub_ixs = nbu.remove_dupes(sub_ixs, w, c)
    grad_update_jit(W, sub_ixs, eta)
    
def inner_update(W, negsamps, eta, w=0, c=0):
    #print(negsamps, eta, w, c)
    if (w in negsamps) or (c in negsamps):
        negsamps = [x for x in negsamps if x not in {w, c}]
    sub_ixs = np.array([w, c] + negsamps) # list(negsamps)
    grad_update(W, sub_ixs, eta)
    
def check_padding(sampler, k, grad_update, dims=1):
    "Poor dependent type checker"
    _samp = next(sampler)[0] if dims == 2 else next(sampler)
    k_ = len(_samp)
    padded = k_ == 2 + k
    assert k_ == sampler.pad + sampler.K
    assert k == sampler.K
    assert k_ in (2 + k, k), 'Length of samples should be either `k` or `k` + 2 if padded'
    assert padded == (grad_update in padded_updates), 'Make sure sampler size agrees with grad_update'   

padded_updates = {grad_update_jit_pad}


gen_npl = NegSampler(neg_sampler_np, toki, K=cnf_.K, ret_type=list)
numpy_opts = dict(ns_grad_=ns_grad, neg_sampler=gen_npl, sliding_window=sliding_window, grad_update=inner_update)

ngsamp_pad = NegSampler(neg_sampler_jit_pad, toki, cfp.K, ret_type=np.ndarray, pad=2)
fast_opts = dict(ns_grad_=ns_grad_jit, neg_sampler=ngsamp_pad, sliding_window=sliding_window_jit, grad_update=grad_update_jit_pad)


# And finally, here is the full gradient descent function put together.

# In[ ]:

def sgd(W=None, corp=None, cf={}, ns_grad_=ns_grad, neg_sampler=None,
        vc=None, sliding_window=sliding_window, grad_update=grad_update_jit_pad):
    check_padding(neg_sampler, cf.K, grad_update)
    if not os.path.exists(cf.dir):
        os.mkdir(cf.dir)
    st = time.time(); cf = Conf(cf)  #.copy()
    assert cf.N == W.shape[1] / 2, 'shape of W disagrees with conf'
    
    iter_corpus = corp[cf.iter:] if cf.iter else corp
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
        for c, negsamps in izip(cont, negsamp_lst):
            grad_update(W, negsamps, eta, w=w, c=c)

    tdur = time.time() - st
    cf2 = update(cf, iter=i+1)  # , norms=norms, gradnorms=gradnorms
    cf2['accumsec'] += tdur
    if not cf2.term and 0:
        fn = join(cf2.dir, 'n{}_e{}.csv'.format(cf.N, cf.epoch))
        DataFrame(W, index=vc).to_csv(fn)
        cf2['epoch'] += 1
        cf2 = update(cf2, iter=0)
    else:
        pass
    #     print(i, 'iters')
    return W, cf2

toka = np.array(toki)
_ = sgd(W=We.copy(), corp=toki, cf=update(cnf_, term={'iters': 1000}), **fast_opts)


# In[ ]:

def mk_sgd2(cf=None, sampler=None):
    iters = cf.term.get('iters') or 0
    
    @nopython
    def loop(W, toka, eta=None, min_eta=None, C=None, K=None, pad=None):
        N = iters or len(toka)
        sub_ixs = np.empty(K + pad, np.int64)
        etas = np.linspace(eta, min_eta, N)
        for i in xrange(N):
            w, conts = sliding_window_inplace(toka, i, C=C)
            eta_ = etas[i]
            for c in conts:
                sampler(sub_ixs)
                grad_update_jit_pad(W, sub_ixs, eta_, w=w, c=c)
        return W
    kwa = z.keyfilter(lambda x: x in 'eta min_eta C K pad'.split(), cf)
    looper = partial(loop, **kwa)
    return looper


a, inplace_sampler = neg_sampler_inplace(toka, cfp.K, pow=.75, pad=cfp.pad, ret_type=np.ndarray)

sgd2 = mk_sgd2(cfp, sampler=inplace_sampler)
# loop(w1, toka[:100], eta=cnf.eta, min_eta=cnf.min_eta, C=cnf.C, K=cnf.K, pad=2)
w1 = We.copy()
sgd2(w1, toka[:100]); None
# sgd2(w1, toka[:100], cfp, sampler=inplace_sampler)


# ### Benchmarks

# In[ ]:

- line profiler
- create array
- index with list


# In[ ]:

_w1, _w2 = We.copy(), We.copy()
tm = {'iters': 100000}
tm = {}


# In[ ]:

cnf_


# In[ ]:

cfp


# In[ ]:

_w1 = We.copy()
get_ipython().magic('time _w1a, _ = sgd(W=_w1, corp=toki, cf=update(cnf_, term=tm), **fast_opts)')
np.linalg.norm(_w1a)


# In[ ]:

_w2 = We.copy()
get_ipython().magic('time _w2a = sgd2(_w2, toka)')
np.linalg.norm(_w2a)


# In[ ]:




# In[ ]:

get_ipython().system('say done')


# In[ ]:

_wpd = We.copy()
get_ipython().magic('prun -q -D prof/sgd.profile _wpd, _ = sgd(W=_wpd, corp=toki, cf=update(cnf, term=tm), **fast_opts)')


# In[ ]:

_wpd = We.copy()
fast_opts2 = update(fast_opts, sliding_window=sliding_window_jit_arr)
get_ipython().magic('time _wpd, _ = sgd(W=_wpd, corp=toka, cf=update(cnf, term=tm), **fast_opts2)')


# ### Gensim

# In[ ]:

import gensim
from gensim.models.word2vec import Word2Vec


# In[ ]:

gparams = ut.to_gensim_params(cnf_)


# In[ ]:

ut.reset_gensim(slow=True, gensim=gensim)
assert gensim.models.word2vec.FAST_VERSION == -1


# In[ ]:

get_ipython().magic('time gen_slow = Word2Vec(brown.sents(), **gparams)')


# In[ ]:

ut.reset_gensim(slow=False, gensim=gensim)
assert gensim.models.word2vec.FAST_VERSION == 1


# In[ ]:

get_ipython().magic('time gen_fast = Word2Vec(brown.sents(), **gparams)')


# In[ ]:

1


# In[ ]:

get_ipython().system('say done')


# # TODO: ~
# [#Begin](#Begin)

# In[ ]:

_w1, _wpd = We.copy(), We.copy()
for i in range(8):
    _w1, _ = sgd(W=_w1.copy(), corp=toki, cf=update(cnf_, term={'iters': 10000}), **numpy_opts) 
    _wpd, _ = sgd(W=_wpd.copy(), corp=toki, cf=update(cnf_, term={'iters': 10000}), **fast_opts)
    
    n1, n2 = np.linalg.norm(_w1), np.linalg.norm(_wpd)
    print('{:.2f}; {:.2f}'.format(n1, n2))


# In[ ]:

w1 = We.copy()
sgd2(w1, toka[:100], cfp, sampler=inplace_sampler)
sgd(W=We.copy(), corp=toki, cf=update(cnf, term={'iters': 1000}), **fast_opts)


# In[ ]:

# tm = {'iters': 10000}
tm = {}
n = 10000
_w1, _wpd = We.copy(), We.copy()

for i in range(8):
    _w1 = sgd2(_w1, toka[:n], cf=update(cfp, term=tm), sampler=inplace_sampler)
    _wpd, _ = sgd(W=_wpd, corp=toki[:n], cf=update(cnf_, term=tm), **fast_opts)
    
    n1, n2 = np.linalg.norm(_w1), np.linalg.norm(_wpd)
    print('{:.2f}; {:.2f}'.format(n1, n2))


# In[ ]:

_ = sgd(W=We.copy(), corp=toki, cf=update(cnf, term={'iters': 10000}), **fast_opts) 


# In[ ]:

_ = sgd(W=We.copy(), corp=toki, cf=update(cnf, term={'iters': 1000}), **fast_opts2) 


# In[ ]:

_ = sgd(W=We.copy(), corp=toki, cf=update(cnf, term={'iters': 10000}), **fast_opts) 


# In[ ]:

sgd(W=We.copy(), corp=toki, cf=update(cnf, term={'iters': 10000}), **numpy_opts)


# In[ ]:

get_ipython().magic('lprun -T lp5.txt -s -f sgd sgd(W=We.copy(), corp=toki, cf=update(cnf, term=trm), **numpy_opts)  # inner_update')


# In[ ]:

trm = {}
# trm = {'iters': 100000}


# In[ ]:

get_ipython().magic('lprun -T lp6.txt -s -f sgd sgd(W=We.copy(), corp=toki, cf=update(cnf, term=tm), **fast_opts)')
# %lprun -T lp7.txt -s -f sgd sgd(W=We.copy(), corp=toki, cf=update(cnf, term=tm), **fast_opts) 


# In[ ]:

get_ipython().magic('lprun -T lp8.txt -s -f sgd2 sgd2(W=We.copy(), corp=toki, cf=update(cnf, term=tm), **fast_opts3)')


# In[ ]:

8.5 -> 12.5
     


# In[ ]:

3.416416      8.6     35.5      grad = ns_grad_jit(Wsub)
6.131119     15.4     93.2      grad_update_jit(W, sub_ixs, eta)


# In[ ]:

get_ipython().magic('time res = sgd(W=We.copy(), corp=toki, cf=update(cnf, term={}), **fast_opts)')


# In[ ]:

get_ipython().magic('time res = sgd(W=We.copy(), corp=toka, cf=update(cnf, term={}), **fast_opts)')


# In[ ]:




# In[ ]:

_ = sgd(W=We.copy(), corp=toki, cf=update(cnf, term={'iters': 10000}), **fast_opts)


# In[ ]:

sgd(W=We.copy(), corp=toki, cf=update(cnfe, term={'iters': 10000}), **kw)


# In[ ]:

get_ipython().magic("lprun -T lp.txt -s -f sgd sgd(W=We.copy(), corp=toki, cf=update(cnfe, term={'iters': 10000})) # ls[:20]")


# rand_ixs = lambda W, n=8, axis=0: nr.randint(0, W.shape[axis], size=n)

# In[ ]:

for i in range(20):
    print('Epoch {}'.format(cnfe.epoch))
    We2, cnfe = sgd(W=We.copy(), corp=toki, cf=update(cnfe, term=dict()), **fast_opts)
    break


# ## Corpus

# In[ ]:

import nltk; reload(nltk)
from gensim.models import Word2Vec


# In[ ]:

ilen(brown.words())


# In[ ]:

brown.


# In[ ]:

ilen(reuters.words())


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




# In[ ]:

Word2Vec(sen)


# In[ ]:

def ev(mod):
    ans = mod.accuracy('src/questions-words.txt', restrict_vocab=10000)
    sect = [d for d in ans if d['section'] == 'total']
    return sum([1 for d in sect for _ in d['correct']])


# In[ ]:

cnf


# In[ ]:




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

cnff = Conf(from_gensim_params(gparams, cnf, dir='cache/v13'))
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

get_ipython().magic('time modf.run()')


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

toki = 


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
#          {\partial \boldsymbol v_{w_j}^{\prime T} \boldsymbol h}
#          = \sigma(\boldsymbol v_{w_j}^{\prime T} \boldsymbol h) -t_j
# $$

# # Analyze word vectors

# In[ ]:

z

