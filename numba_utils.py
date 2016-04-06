from builtins import range as xrange
from numba import njit
import numpy as np
import numpy.random as nr

getNall = njit(lambda W: W.shape[1] // 2)
sig = njit(lambda x: 1 / (1 + np.exp(-x)))
gen_labels = njit(lambda pos_negs: [1] + [0] * (len(pos_negs) - 1))


# Gradient calculation
@njit
def dot(a, b):
    sm = 0
    for i in range(len(a)):
        sm += a[i] * b[i]
    return sm


@njit
def get_vecs1(Wsub):
    """Extract weight vectors from subset of
    negative-sampling skip-gram weight matrix.
    First row is input vector, second row is outout vector,
    remaining rows are negative iutput vectors"""
    length = len(Wsub)
    N = getNall(Wsub)
    h = Wsub[0, :N]  # ∈ ℝⁿ
    vwo_negsamps = Wsub[1:length, N:]
    return h, vwo_negsamps


@njit
def ns_prob(h=None, vout=None, label=None):
    return sig(dot(vout, h)) - label


@njit
def ns_loss_grads(h, vout, label):
    dotprod = ns_prob(h=h, vout=vout, label=label)
    return dotprod * vout, dotprod * h


@njit
def inner(Wsub_grad, N, h, i, vout, label):
    hgrad, vgrad = ns_loss_grads(h, vout, label)
    for j in range(N):
        Wsub_grad[0, j] += hgrad[j]
        Wsub_grad[i, N + j] += vgrad[j]


@njit
def loop(Wsub_grad, N, h, vwo_negsamps):
    for i, label in enumerate(gen_labels(vwo_negsamps), 1):
        inner(Wsub_grad, N, h, i, vwo_negsamps[i - 1], label)


@njit
def ns_grad(Wsub):
    h, vwo_negsamps = get_vecs1(Wsub)
    N = getNall(Wsub)
    Wsub_grad = np.zeros(Wsub.shape)
    loop(Wsub_grad, N, h, vwo_negsamps)
    return Wsub_grad


# Negative sampler
@njit
def bisect_left_jit(a, v):
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


@njit
def neg_sampler_jitl_(cum_prob, K):
    while 1:
        l = []
        for i in range(K):
            l.append(bisect_left_jit(cum_prob, nr.rand()))
        yield l

@njit
def neg_sampler_jita_(cum_prob, K):
    while 1:
        a = np.empty(K, dtype=np.int64)
        for i in range(K):
            a[i] = bisect_left_jit(cum_prob, nr.rand())
        yield a

@njit
def neg_sampler_jitl_pad(cum_prob, K, pad=0):
    init = [0] * pad
    while 1:
        l = list(init)
        for i in range(K):
            l.append(bisect_left_jit(cum_prob, nr.rand()))
        yield l

@njit
def neg_sampler_jita_pad(cum_prob, K, pad=0):
    while 1:
        a = np.empty(K + pad, dtype=np.int64)
        for i in range(pad, K + pad):
            a[i] = bisect_left_jit(cum_prob, nr.rand())
        yield a


@njit
def remove_dupes(negsamps: 'array[int]', w, c):
    """Put w, c in negsamps[0,1] negsamps[2:]. If w or c occur in
    negsamps[2:], remove and return results.
    Same behavior as `remove_dupes_`
    """
    num_dupes = count_occ(negsamps, w, start=2) + count_occ(negsamps, c, start=2)
    if not num_dupes:
        negsamps[0], negsamps[1] = w, c
        return negsamps
    L = len(negsamps)
    ns2 = np.empty(L - num_dupes, dtype=np.int64)
    ns2[0], ns2[1] = w, c

    newj = 2
    for i in range(2, L):
        if (negsamps[i] == w) or (negsamps[i] == c):
            continue
        ns2[newj] = negsamps[i]
        newj += 1
    return ns2

remove_dupes_ = lambda negsamps, w, c: np.array([w, c] + [n for n in negsamps[2:] if n not in (w, c)])


# Array funcs
@njit
def concat_jit(arr, *xs):
    X = len(xs)
    A = len(arr)
    a2 = np.empty(A + X, dtype=np.uint32)
    for i in xrange(X):
        a2[i] = xs[i]

    for i in xrange(X, A + X):
        a2[i] = arr[i - X]
    return a2

@njit
def concat(a, b):
    na = len(a)
    n = na + len(b)
    c = np.empty(n, dtype=a.dtype)
    for i in range(na):
        c[i] = a[i]
    for i in range(len(b)):
        c[i + na] = b[i]
    return c


@njit
def count_occ(arr, x, start=0):
    "# of occurrences of `x` in `arr`"
    ct = 0
    for i in range(start, len(arr)):
        if x == arr[i]:
            ct += 1
    return ct


@njit
def contains(arr, x):
    for e in arr:
        if x == e:
            return True
    return False


#Sliding window
@njit
def bounds_check_window(i, xs: [int], winsize, N):
    x = xs[i]
    ix1 = max(0, i-winsize)
    ix2 = min(N, i+winsize+1)
    return x, xs[ix1:i] + xs[i + 1:ix2]


@njit
def bounds_check_window_arr(i, xs: np.array, winsize, N):
    x = xs[i]
    ix1 = max(0, i-winsize)
    ix2 = min(N, i+winsize+1)
    return x, concat(xs[ix1:i], xs[i + 1:ix2])


@njit
def nseed(x):
    return nr.seed(x)


# Eval
@njit
def sum0(arr):
    m, n = arr.shape
    a = np.zeros(n)
    for i in xrange(m):
        for j in xrange(n):
            a[j] += arr[i, j]
    return a

@njit
def combine_(vecs):
    v = sum0(vecs) / len(vecs)
    return v / norm_jit1d(v)


@njit
def norm_jit1d(a):
    return np.sqrt(np.sum(a ** 2))


@njit
def norm_jit2d(a):
    sm = 0
    n, m = a.shape
    for i in xrange(n):
        for j in xrange(m):
            sm += a[i, j] ** 2
    return np.sqrt(sm)


@njit
def ix_combine_(a, ixs, nplus):
    vecs = a[ixs]
    vecs[nplus:, :] *= -1
    return combine_(vecs)