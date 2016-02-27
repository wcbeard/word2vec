from wordvec_utils import WordVectorizer, UNKNOWN, get_window, Cat, get_win, get_rand_wins, get_wins
from itertools import islice
import numpy.random as nr


def test_WordVectorizer():
    txt = "I am going to the store".split()
    dv = WordVectorizer().fit([txt])
    txt2 = "I am going to the shed".split()

    csr = dv.transform(txt2)
    assert dv.inverse_transform(csr, transform=True) == ['I', 'am', 'going', 'to', 'the', UNKNOWN]
    assert len(txt2) == csr.shape[0], 'Number of output rows should be len of input'
    assert len(txt2) == len(csr.indices), 'Number of nonzero entries should be len of input'
    return dv


def test_window():
    txt = 'You will rejoice to hear that no disaster has accompanied the commencement'.split()
    a, b = islice(get_window(txt, 4), 2)
    assert a == (['You', 'will', 'rejoice', 'to'], 'hear', ['that', 'no', 'disaster', 'has'])
    assert b == (['will', 'rejoice', 'to', 'hear'], 'that', ['no', 'disaster', 'has', 'accompanied'])


def test_get_win():
    C = 4
    tl = 'Project Gutenberg s Frankenstein by Mary Wollstonecraft Godwin Shelley This'.split()
    assert get_win(tl, 4, C=C) == ('by', ['Project', 'Gutenberg', 's', 'Frankenstein', 'Mary', 'Wollstonecraft', 'Godwin', 'Shelley'])
    assert get_win(tl, 5, C=C) == ('Mary', ['Gutenberg', 's', 'Frankenstein', 'by', 'Wollstonecraft', 'Godwin', 'Shelley', 'This'])
    # assert get_win(tl, 4, C=C) == (['Project', 'Gutenberg', 's', 'Frankenstein'], 'by', ['Mary', 'Wollstonecraft', 'Godwin', 'Shelley'])
    # assert get_win(tl, 5, C=C) == (['Gutenberg', 's', 'Frankenstein', 'by'], 'Mary', ['Wollstonecraft', 'Godwin', 'Shelley', 'This'])


def test_get_rand_win():
    C = 4
    tl = 'Project Gutenberg s Frankenstein by Mary Wollstonecraft Godwin Shelley This'.split()
    assert {w for w, _ in islice(get_rand_wins([''] * 4 + tl + [''] * 4, C=4, seed=1), 100)} == set(tl)


def test_uncat():
    N, V = 5, 1000
    nr.seed(0)
    W1 = nr.randn(V, N) * .01  #sp.sparse.csr_matrix()
    W2 = nr.randn(N, V) * .01  #sp.sparse.csr_matrix()
    w1a, w1b = Cat.split(Cat.join(W1, W2))
    assert (w1a == W1).all()
    assert (w1b == W2).all()


def test_get_wins():
    l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert get_wins(4, l, winlen=2, cat=0) == ([2, 3], [5, 6])
    assert get_wins(1, l, winlen=2, cat=0) == ([0], [2, 3])
    assert get_wins(9, l, winlen=2, cat=0) == ([7, 8], [10])
    assert get_wins(9, l, winlen=2, cat=1) == [7, 8, 10]