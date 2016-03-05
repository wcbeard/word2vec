from numba import jit
import numpy as np

nopython = jit(nopython=1)
getNall = nopython(lambda W: W.shape[1] // 2)
sig = nopython(lambda x: 1 / (1 + np.exp(-x)))
gen_labels = nopython(lambda pos_negs: [1] + [0] * (len(pos_negs) - 1))


@nopython
def dot(a, b):
    sm = 0
    for i in range(len(a)):
        sm += a[i] * b[i]
    return sm


@nopython
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


@nopython
def ns_prob(h=None, vout=None, label=None):
    return sig(dot(vout, h)) - label


@nopython
def ns_loss_grads(h, vout, label):
    dotprod = ns_prob(h=h, vout=vout, label=label)
    return dotprod * vout, dotprod * h


@nopython
def inner(Wsub_grad, N, h, i, vout, label):
    hgrad, vgrad = ns_loss_grads(h, vout, label)
    for j in range(N):
        Wsub_grad[0, j] += hgrad[j]
        Wsub_grad[i, N + j] += vgrad[j]


@nopython
def loop(Wsub_grad, N, h, vwo_negsamps):
    for i, label in enumerate(gen_labels(vwo_negsamps), 1):
        inner(Wsub_grad, N, h, i, vwo_negsamps[i - 1], label)


@nopython
def ns_grad(Wsub):
    h, vwo_negsamps = get_vecs1(Wsub)
    N = getNall(Wsub)
    Wsub_grad = np.zeros(Wsub.shape)
    loop(Wsub_grad, N, h, vwo_negsamps)
    return Wsub_grad