import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold


dfs = ( pd.read_csv('offline_challenge/xtrain_obfuscated.txt', header=None, names=['text']),
        pd.read_csv('offline_challenge/ytrain.txt',header=None, names=['label']))
df = pd.concat(dfs,axis=1)
labels =pd.read_csv('labels.txt',header=None, names=['label','title'], sep='\t')
df = df.merge(labels, on='label')
df['len']=df.text.str.len()
# df.groupby('title')['len'].hist(normed=True)
# normed doesn't seem to work
count_vec = CountVectorizer(analyzer='char', ngram_range=(3,6))
text = count_vec.fit_transform(df.text)
voc = pd.Series(count_vec.vocabulary_).sort_values()
# text.sdf = pd.SparseDataFrame(text, columns=voc.index)

voc = pd.DataFrame(voc)
voc.columns=['ind']

cnts=text.sum(axis=0)
one = text.getnnz(axis=0)
# text.max() is 84


voc['counts'] = cnts.T
voc['ones'] = one
voc['ngram'] = voc.index.str.len()

def vw_create(filename, labels, csr_mat, columns, selection):
    with open(filename,'w') as f:
        row_start=0
        for i_row, label in enumerate(labels):
            row_end = text.indptr[i_row]
            inds = csr_mat.indices[row_start:row_end]
            dats = csr_mat.data[row_start:row_end]

            f.write('{} | '.format(label+1))

            for i_col in inds:
                if selection(i_col):
                    f.write('{}:{}')

            row_start = row_end

# a = voc.loc[voc.ngram==1,'counts'].sort_values(ascending=False)
# b = voc.loc[voc.ngram==2,'counts'].sort_values(ascending=False)
# c = voc.loc[voc.ngram==3,'counts'].sort_values(ascending=False)

# a1 = voc.loc[voc.ngram==1,'ones'].sort_values(ascending=False)
# b1 = voc.loc[voc.ngram==2,'ones'].sort_values(ascending=False)
# c1 = voc.loc[voc.ngram==3,'ones'].sort_values(ascending=False)

#(voc['ones']>10).sum() 56591
# voc.shape (133584, 4)


#text_std = text.std(axis=0)
logreg = LogisticRegression(solver='lbfgs',verbose=2)

pipe_base = Pipeline([('std',StandardScaler()), ('logreg',logreg)])
pipe_raw = Pipeline([ ('logreg',logreg)])

params={'logreg__C' :np.power(10.0,np.arange(-5,5))}


skf = StratifiedKFold(n_splits=5, random_state=1234)
gs1 = GridSearchCV(estimator= pipe_raw, param_grid= params, scoring='neg_log_loss',n_jobs=1, cv=skf)
gs1.fit(text, df.label)
with open('gs1','w') as f:
    pickle.dump(gs1,f)

with open('gs1','w') as f:
    pickle.dump(gs1, f)



        # todo
# use ones rather than counts
# use normalisation
# use interaction terms >10 ones
# df.text

# df.text.str

# df.groupby('label').size()