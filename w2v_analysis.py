
# coding: utf-8

# In[ ]:

get_ipython().run_cell_magic('javascript', '', "var csc = IPython.keyboard_manager.command_shortcuts\ncsc.add_shortcut('Ctrl-k','ipython.move-selected-cell-up')\ncsc.add_shortcut('Ctrl-j','ipython.move-selected-cell-down')\ncsc.add_shortcut('Shift-m','ipython.merge-selected-cell-with-cell-after')")


# In[ ]:

import warnings; warnings.filterwarnings('ignore')
from py3k_imports import *
import project_imports3; reload(project_imports3); from project_imports3 import *

get_ipython().magic('matplotlib inline')
pu.psettings(pd)
pd.options.display.width = 200  # 150


# In[ ]:

import utils as ut


# In[ ]:

from collections import Counter
import math
import wordvec_utils; reload(wordvec_utils); from wordvec_utils import *


# with open('cache/dv_les', 'r') as f:
#     txt = f.read().splitlines()

# tx = np.array([t.strip("'") for t in map(str.lower, re.split(r'[^A-Za-z\']', txt)) if t])
# txsub = get_subsample(tx)

# freqdf = inspect_freq_thresh(tx)

# p = get_subsample_prob(tx, .001)
# plt.hist(p, bins=100); None

# ## Read cached vectors

# In[ ]:

def read_vecs(ws=[], dct=False, prefdir='cache/v2', n=25):
    fn_pat = join(prefdir, r'n%s_e{}.csv' % n)
    fn_re = re.compile(fn_pat.format(r'(\d+)'))

    # accums = sorted(map(z.comp(int, itg(0), fn_re.findall), glob.glob(fn_pat.format('*'))), reverse=1)
    accums = {z.comp(int, itg(0), fn_re.findall)(fn): fn for fn in glob.glob(fn_pat.format('*'))}
    if dct:
        return {sec: pd.read_csv(fn, index_col=0, dialect=print(fn)).rename(columns=int)
                for sec, fn in accums.items()}
    Ws = [pd.read_csv(fn, index_col=0, dialect=print(fn)).rename(columns=int) for i, (_, fn)
          in enumerate(sorted(accums.items())) if i >= len(ws)]
    return ws + Ws


# In[ ]:

Ws = []


# In[ ]:

class WVecArith(object):
    def __init__(self, W=None):
        self.W = W
        self.ixs = list(W.index)
        
    def __getattr__(self, attr):
        return self.W.ix[attr]
    
    def __dir__(self):
        return self.ixs


with open('txt.txt','r') as f:
    txt = f.read().splitlines()

freq = Counter(txt)

def keep_freqs(W, freq=freq, min_count=5):
    new_ix = [x for x in W.index if freq[x] >= min_count]
    return W.ix[new_ix].copy()


# In[ ]:

Ws = read_vecs(ws=Ws, prefdir='cache/v11/', n=100)
W = Ws[-1]
print(len(Ws))
phs = lambda: None
phs.__dict__.update(dict(zip(W.index, W.index)))
w = WVecArith(W=W)
Wf = keep_freqs(W, freq=freq, min_count=5)


# In[ ]:




# In[ ]:

W[:2]


# ## Eval

# In[ ]:

W[60:62]


# In[ ]:

def get_closest(wd='death', n=20, W=None):
    if isinstance(wd, str):
        wvec = W.ix[wd]
    else:
        wvec = wd
    dst = cdist(wvec, W)
    global closests
    closests = dst.sort_values(ascending=True)[:n]
    #print(closests)
    cvecs = W.ix[closests.index]
    df = DataFrame(OrderedDict([#('Freq', freq.ix[dv.wds(closests)]),
                                ('Dist', dst.ix[closests.index]),
                                ('Size', np.diag(cvecs @ cvecs.T)),
                                ('Freq', closests.index.map(freq.get)),
            ]))
    df.Dist = df.Dist.map('{:.2f}'.format)
    df.Size = df.Size.map('{:.1f}'.format)
    # df.Freq = df.Freq.round()
    
    # get_closest(W=w)
    return df.reset_index(drop=0).rename(columns={'index': 'Word'})

def get_closests(wds, n=20, W=None, fna=''):
    df = pu.side_by_side([get_closest(wd=w, n=n + 1, W=W) for w in wds], names=wds) #.fillna(fna)
    # Make sure closest word is self, then drop it
    for c, _ in df:
        assert df[(c, 'Word')][0] == c, 'Closest word must be self'
        assert float(df[(c, 'Dist')][0]) == 0, 'Closest word must be self, distance=0'
    return df[1:].reset_index(drop=1)

Series.__matmul__ = Series.dot
DataFrame.__matmul__ = DataFrame.dot


# In[ ]:

Series(list(freq.values())).value_counts(normalize=1).sort_values(ascending=True).cumsum()


# In[ ]:

get_closest(W.ix['Ron_Weasley'] - W.ix['Ron'] + W.ix['Arthur'], n=20, W=Wf)


# In[ ]:

harry_friends = ['Hermione_Granger', 'Ron_Weasley', 'Harry_Potter', ]
harry_friends = ['Hermione', 'Ron', 'Harry', ]
harry_friends = [phs.Hermione_Granger, phs.Ron_Weasley, phs.Harry_Potter, ]
W.ix[harry_friends]


# In[ ]:

get_closest(w.Killing_Curse - w.Bellatrix_Lestrange + w.Harry, n=50, W=W)


# In[ ]:

get_closest(w.Avada_Kedavra - w.Voldemort + w.Ron, n=50, W=W)


# In[ ]:

EXPELLIARMUS
Avada
Tergeo


# In[ ]:

get_closest((w.Wingardium), n=50, W=W)


# In[ ]:




# In[ ]:

w.Expelliarmus + w.Petrificus_Totalus + w.Stu


# In[ ]:

get_closest(w.Expelliarmus - w.EXPELLIARMUS + w.Expecto, n=50, W=Wf)


# In[ ]:

get_closest(w.STUPEFY, n=50, W=Wf)


# In[ ]:

# get_closest(w.Avada_Kedavra - w.Voldemort + w.Harry, n=50, W=W)  # => Expelliarmus
# get_closest(-(w.Stupefy) + w.Expelliarmus + w.Harry, n=50, W=W)  # => Avada_Kedavra
get_closest(-(w.Disarmed) + w.Expelliarmus + w.Harry, n=50, W=Wf)  # => Expelliarmus
get_closest(w.Stupefy, n=50, W=Wf)


# In[ ]:

get_closest(w.Tonks - w.Nymphadora + w.Remus, n=50, W=W)
get_closest(w.Umbridge - w.investigate + w.Dumbledore, n=90, W=W)
get_closest(w.Dumbledore - w.kindly + w.Umbridge, n=90, W=W)  # => hastily, dangerously


# In[ ]:

get_closest(w.Bill - w.Fleur_Delacour + w.Nymphadora, n=50, W=W)


# In[ ]:

Wingardi


# In[ ]:

get_closest(w.Kingsley - w.Aurors + w.Death_Eaters, n=50, W=W)


# In[ ]:

# get_closest(W.ix['Hermione_Granger'] - W.ix['Hermione'] + W.ix['Hermione'], n=20, W=W)
get_closest(w.Hermione_Granger - w.Hermione + w.Hermione, n=20, W=W)


# In[ ]:

get_closest(W.ix[phs.Crabbe] - W.ix[phs.Malfoy] + W.ix[phs.Harry], n=20, W=W)
get_closest(W.ix[phs.Lucius] - W.ix[phs.Malfoy] + W.ix[phs.Ron], n=20, W=W)
# get_closest(W.ix['You-Know-Who'], n=90, W=W)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

get_closests(harry_friends, n=40, W=W)


# In[ ]:

swap level, get word


# In[ ]:

Dolores Jane Umbridge


# In[ ]:

Professor Umbridge


# In[ ]:

for i, tk in enumerate(txt):
#     if tk == 'salmon-pink':
    if 'expecto' in tk.lower():
        print(' '.join(txt[i-10:i+10]))


# In[ ]:

get_closests(wds, W=Ws[-1])


# ## Phrases

# In[ ]:

phrases = [x for x in Ws[0].index if '_' in x and "'" not in x]
phrases_pos = [x for x in Ws[0].index if '_' in x and "'" in x]  # possessive


# In[ ]:

with open('phrases.txt', 'w') as f:
    f.write('\n'.join(phrases))


# In[ ]:

norms = W.T.apply(np.linalg.norm).T
# norms = W.ix[phrases].T.apply(np.linalg.norm).T


# In[ ]:

norms.sort_values(ascending=False)[:30]


# In[ ]:

sns.clustermap(d1[:], annot=True, fmt=".0f", figsize=(20, 12));


# In[ ]:

sns.clustermap(Ws[2].ix[fives].T, annot=True, fmt=".0f", figsize=(20, 7));


# In[ ]:

sns.clustermap(Ws[-1].ix[all_nums_wds].T, annot=True, fmt=".0f", figsize=(25, 10));


# ## Prob words

# In[ ]:

from spacy.en import English
nlp = English()


# In[ ]:

parsed = nlp((' '.join(set(txt))))
probs = {w.orth_: w.prob for w in parsed}
get_ipython().system('say done')


# In[ ]:

prbdf = Series(probs).reset_index(drop=0).rename(columns={'index': 'Word', 0: 'Prob'})
prbdf['Freq'] = prbdf.Word.map(tct)
prbdf = prbdf.sort_values(['Prob', 'Freq'], ascending=[1, 0])
prbdf = prbdf[prbdf.Word.str[0].str.isupper() & ~prbdf.Word.str[-1].str.isupper()].query('Freq > 30').reset_index(drop=1)


# In[ ]:




# In[ ]:

n = 80
prbdf[n:n+80]


# # Linear

# In[ ]:

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('log.csv', sep='\t', index_col=0).reset_index(drop=1).rename(columns=str.capitalize)
df.loc[df.Sample == 0, 'Sample'] = .17
df['Size_10'] = df.Size // 10 * 10
# df = df.query('Sample > 0.00001')
xcols = list(df.columns[1:])

rf = RandomForestRegressor()
lsr = LinearRegression(normalize=1)
lr = Lasso(normalize=1)
el = ElasticNet(normalize=1)
poly = PolynomialFeatures(degree=2, interaction_only=0)

X = df[xcols].values
X2 = poly.fit_transform(X)
y = df.Score.values
y0 = y == 0
del y0
# yg = y == 0


# In[ ]:




# In[ ]:

df.Sample.value_counts(normalize=0)


# In[ ]:

df.groupby('Sample').Score.agg(['mean', 'median'])


# In[ ]:

plt.figure(figsize=(16, 10))
sns.violinplot(x="Sample", y="Score", data=df.query('Sample != [0.00001, 0.00010]'))


# In[ ]:

dfa = df.query('Sample != [0.00001, 0.00010] & Size_10 > 50')
dfa.groupby(['Sample', 'Size_10', 'Window', 'Negative']).Score.agg(['mean', 'median', 'size']).round(2).to_csv('/tmp/st.csv')


# In[ ]:

get_ipython().system("open '/tmp/st.csv'")


# In[ ]:




# In[ ]:

df.pivot()


# In[ ]:




# In[ ]:

import statsmodels


# In[ ]:

reload(sns)


# In[ ]:

plt.figure(figsize=(16, 10))
sns.interactplot('Size', 'Window', 'Score', data=df)
# sns.interactplot('Size', 'Sample', 'Score', data=df)


# In[ ]:

plt.figure(figsize=(16, 10))
d2 = df.groupby(['Size_10', 'Window']).Score.mean().unstack()
sns.heatmap(d2, annot=True, fmt=".0f", linewidths=.5)


# In[ ]:

plt.figure(figsize=(16, 10))
d2 = df.groupby(['Window', 'Negative']).Score.mean().unstack()
sns.heatmap(d2, annot=True, fmt=".0f", linewidths=.5)


# In[ ]:

negsamp=7, N=100, window=4


# In[ ]:

plt.figure(figsize=(16, 10))
d1 = df.groupby(['Size_10', 'Sample']).Score.mean().unstack()
sns.heatmap(d1, annot=True, fmt=".0f", linewidths=.5)


# In[ ]:

plt.figure(figsize=(16, 10))
sns.interactplot('Size', 'Window', 'Score', data=df)


# In[ ]:

get_ipython().magic('pinfo sns.interactplot')


# In[ ]:

sns.heatmap(flights, annot=True, fmt="d", linewidths=.5)


# In[ ]:

df.query('Sample == .00').sort_values('Score', ascending=0)[:40]


# In[ ]:

df.sort_values('Score', ascending=0)[:40]


# In[ ]:

df.query('Score < 10').sort_values('Score', ascending=True)


# In[ ]:

df.Score.hist(bins=50)


# In[ ]:

def poly_labels(powers, cols=xcols):
    pow = lambda x: '^{}'.format(x) if x > 1 else ''
    return '_'.join(x + pow(bool) for x, bool in zip(cols, powers) if bool) or 'None'


# In[ ]:

def mk_feat_df(X, y, cols=xcols):
    lr.fit(X, y)
    lsr.fit(X, y)
    rf.fit(X, y)
    el.fit(X, y)
    return DataFrame([rf.feature_importances_, lsr.coef_, el.coef_, lr.coef_],
                     columns=cols, index=['Rf', 'Lasso', 'El', 'Lr', ]).T.round(2)


# In[ ]:

xcols


# mcols = ['Negative', 'Sample', 'Sg', 'Window']
# Xmiri = df[mcols].values
# 
# mk_feat_df(Xmiri, y, cols=mcols)

# In[ ]:

mk_feat_df(X, y)


# In[ ]:

lr.fit(X2, y0)


# In[ ]:

poly_cols = map(poly_labels, poly.powers_)
mk_feat_df(X2, y, cols=poly_cols) #.applymap('{:.1f}'.format)


# In[ ]:

thresh = 160
his = df.query('Score > @thresh & Score > @thresh')
los = df.query('Score <= @thresh')

ax1 = sns.kdeplot(los.Sample, los.Size, cmap="Blues", shade_lowest=False)
ax2 = sns.kdeplot(his.Sample, his.Size, cmap="Reds", shade_lowest=0)
# plt.xlim(-0.05, .05, None)
# ax1.set(xscale="log")
# ax2.set(xscale="log");


# In[ ]:

sns.pairplot(his)


# In[ ]:

df.query('sample == 0.01')


# In[ ]:

df.sort_values(['Sample', 'Size'][::-1], ascending=[0, 0])[:40]


# In[ ]:

lr.coef_


# In[ ]:

np.hstack([rf.feature_importances_, lr.coef_])


# In[ ]:

DataFrame([rf.feature_importances_, lsr.coef_, lr.coef_], columns=df.columns[1:])


# In[ ]:

df[:2]


# In[ ]:

lr.coef_


# In[ ]:




# In[ ]:

lr.residues_


# In[ ]:

df


# ## PCA

# In[ ]:

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
pca = PCA(n_components=2)
tsne = TSNE(n_components=2, random_state=1)


# In[ ]:

def sub(xs, ys):
    yss = set(ys)
    return [x for x in xs if x not in yss]


# In[ ]:

# %matplotlib notebook
get_ipython().magic('matplotlib inline')


# In[ ]:

_sub = 'Grimmauld_Place Romilda_Vane Little_Hangleton Ludo_Bagman Michael_Corner Dedalus_Diggle Marcus_Flint'.split()
_add = """Tonks Nymphadora Remus Lupin Dolores Umbridge Dumbledore's_Army Severus
Greyback Goyle Crabbe Aurors Auror Ginny Albus Dumbledore Snape Ron Hermione Harry Mad-Eye Bill Hogwarts
Kingsley Crucio Expelliarmus Filch Nagini Hedwig
""".split()  # Moody
# phs.Dumbledore's_Army
phrases2 = sub(phrases, _sub) + _add


# In[ ]:

from sklearn.preprocessing import StandardScaler


# # TODO: StandardScaler

# In[ ]:

xydf.ix[['Nagini', 'Hedwig', 'Mrs_Norris']]


# In[ ]:

def plot_pca(i, wds=None, fig=None, figsize=(16, 10), lims=(-1.5, None), fontsize=12, fontnum=0):
    msg = set(wds) - set(W.index)
    assert not msg, msg
    w = Ws[i].ix[wds]
    dimredux = TSNE(n_components=2, random_state=1, perplexity=20, learning_rate=100, angle=.2, early_exaggeration=8) #pca  #tsne
    dimredux = pca
    P = DataFrame(dimredux.fit_transform(w), index=w.index)
    xs, ys = zip(*P.values)
    
    if fig is None:
        plt.figure(figsize=figsize)
    plt.scatter(xs, ys, marker='.', s=25)
    # plt.plot()
    for s, x, y in zip(wds, xs, ys):
        plt.text(x, y, s, horizontalalignment='left', verticalalignment='center')
#     plt.ylim(lims)
    return DataFrame(zip(*[xs, ys]), index=wds)


# In[ ]:

xydf = plot_pca(-1, wds=phrases2, figsize=(20, 15), lims=(None, None))


# In[ ]:

len(zip(*P.values))


# In[ ]:

len(phrases)


# In[ ]:

nr.seed(1)
ph = nr.choice(phrases, size=100, replace=False)


# In[ ]:

def plot_pca3d(i, wds=None, fig=None, figsize=(16, 10), lims=(-1.5, None), fontsize=12, fontnum=0):
    if wds is None:
        wds = fives
    w = Ws[i].ix[wds]
    dimredux = TSNE(n_components=3, random_state=89) #pca  #tsne
    dimredux = PCA(n_components=3)
    P = DataFrame(dimredux.fit_transform(w), index=w.index)
    xs, ys, zs = zip(*P.values)
    
    
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    ax.scatter(xs, ys, zs, marker='.', s=20)
    # plt.plot()
    for s, x, y, z in zip(wds, xs, ys, zs):
        1
        ax.text(x, y, z, s, horizontalalignment='left', verticalalignment='center')
#     plt.ylim(lims)
    return DataFrame(zip(*[xs, ys]))

xydf = plot_pca3d(-1, wds=ph, figsize=(20, 15), lims=(None, None))


# In[ ]:

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# fig = plt.figure()
ax = fig.gca(projection='3d')
 
ax.scatter(xs, ys, zs, c=c, marker=m);


# In[ ]:

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100
for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zl, zh)
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# In[ ]:

w = Ws[-1]


# In[ ]:

lims=(None, 1.5)
lims=(None, None)
xydf = plot_pca(-1, wds=all_nums_wds, figsize=(16, 15), lims=lims, fontsize=12, fontnum=4)


# Wd = read_vecs(dct=1)
# Wds = sorted(Wd)
# plt.plot(*zip(*[(wbk, np.linalg.norm(Wd[wbk] - Wd[wak])) for wak, wbk in zip(Wds[:], Wds[1:])]))
# plt.ylim(0, None);

# Diffsμ = pd.concat([(wb - wa).mean(axis=1) for wa, wb in zip(Ws[1:], Ws[2:])], axis=1)
# Diffsmx = pd.concat([(wb - wa).abs().max(axis=1) for wa, wb in zip(Ws[1:], Ws[2:])], axis=1)

# In[ ]:

Ws = read_vecs(ws=Ws)


# ## Norms

# In[ ]:

plt.plot([np.linalg.norm(wa) for wa in Ws[:]]);


# In[ ]:

plt.plot([np.linalg.norm(wb - wa) for wa, wb in zip(Ws[:], Ws[1:])])


# In[ ]:

plt.plot([np.linalg.norm(wb - wa) / np.linalg.norm(wa) for wa, wb in zip(Ws[:], Ws[1:])])
# plt.ylim(0, None);


# In[ ]:

plt.plot([np.linalg.norm(wb) - np.linalg.norm(wa) for wa, wb in zip(Ws[:], Ws[1:])])


# In[ ]:

:
    print()


# In[ ]:

w = Ws[-1]
w70 = w.ix['seventy'] - w.ix['seven'] + w.ix['eight']
w70 = w.ix['valjean'] - w.ix['man'] + w.ix['woman']
w70 = w.ix['cossette'] - w.ix['woman'] + w.ix['man']
get_closest(wd=w70, n=20, W=w)


# In[ ]:

get_closest(wd='archbishops', n=20, W=w)


# In[ ]:

w = Ws[-1]
w80 = w.ix['seventy'] - w.ix['seven'] + w.ix['five']
w80 = w.ix['fifty'] - w.ix['five'] + w.ix['seven']
get_closest(wd=w80, n=20, W=w)
# w.ix['eighty']


# In[ ]:




# In[ ]:

get_closest(wd='student', n=20, W=w)


# In[ ]:

get


# In[ ]:

sns.clustermap(d1.T)


# In[ ]:




# In[ ]:

get_closests(wds=wds, W=Ws[1], n=30)[('five', )]


# def get_closest(wd='death', n=20, dv=dv, W=W):
#     wvec = W[dv.get(wd)]
#     # dst = dist(wvec, W)
#     dst = cdist(wvec, W)
#     #dst2 = [sp.spatial.distance.cosine(wvec, v) for v in W]
#     closests = np.argsort(dst)[:n]
#     cvecs = W[closests, :]
#     df = DataFrame(OrderedDict([('Freq', freq.ix[dv.wds(closests)]),
#                                 ('Dist', dst[closests]),
#                                 ('Size', np.diag(cvecs @ cvecs.T))]))
#     df.Dist = df.Dist.map('{:.2f}'.format)
#     df.Size = df.Size.map('{:.1f}'.format)
#     df.Freq = df.Freq.round()
#     
#     return df.reset_index(drop=0).rename(columns={'index': 'Word'})
# 
# 
# def get_closests(wds, n=20, dv=dv, W=W, fna='', names=None):
#     df = pu.side_by_side([get_closest(wd=w, n=n + 1, dv=dv, W=W) for w in wds], names=wds) #.fillna(fna)
#     # Make sure closest word is self, then drop it
#     for c, _ in df:
#         assert df[(c, 'Word')][0] == c, 'Closest word must be self'
#         assert float(df[(c, 'Dist')][0]) == 0, 'Closest word must be self, distance=0'
#     return df[1:].reset_index(drop=1)

# In[ ]:

Ws[0]


# In[ ]:

wd1 = pd.read_csv('cache/w')


# In[ ]:

wd1 = pd.read_csv('/tmp/w66860_test.csv', index_col=0)


# In[ ]:




# In[ ]:

Series.__matmul__ = Series.dot
DataFrame.__matmul__ = DataFrame.dot


# In[ ]:

cdist(wd1.ix['wooden'], wd1).order(ascending=0)[:10]


# In[ ]:

cdist(wd1.ix['wood'], wd1).order(ascending=0)[-20:]


# In[ ]:

cdist(wd1.ix['wooden'].values, wd1.values)


# In[ ]:

wd1.ix['wooden']


# ## Eval

# ## Async

# In[ ]:

import asyncio

@asyncio.coroutine
def slow_operation(n):
    yield from asyncio.sleep(10 - n)
    print("Slow operation {} complete".format(n))


@asyncio.coroutine
def main():
    yield from asyncio.wait([
        slow_operation(1),
        slow_operation(2),
        slow_operation(3),
    ])


loop = asyncio.get_event_loop()
loop.run_until_complete(main())


# In[ ]:

w


# In[ ]:

get_ipython().run_cell_magic('time', '', 'import asyncio\n\nasync def slow_operation(n):\n    await asyncio.sleep(5 - n)\n    print("Slow operation {} complete".format(n))\n    return 2 ** n\n\n\nasync def main():\n    await asyncio.wait([\n        slow_operation(1),\n        slow_operation(2),\n        slow_operation(3),\n    ])\n\n\nloop = asyncio.get_event_loop()\nloop.run_until_complete(main())')

