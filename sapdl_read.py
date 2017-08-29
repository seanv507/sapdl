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
df =df.merge(labels, on='label')
df['len']=df.text.str.len()
# df.groupby('title')['len'].hist(normed=True)
# normed doesn't seem to work
count_vec = CountVectorizer(analyzer='char', ngram_range=(1,6))
text = count_vec.fit_transform(df.text)

with open('gs1') as f:
    gs1=pickle.load(f)

./vw -d train.txt -f lg.vw --loss_function logistic
# cache
./vw -d train.txt -c --passes 10


# test and save predictions
./vw -d test.txt -t -i predictor.vw -p predictions.txt

# multiclass Label in {1,2,...,k}
# --oaa k (one against all k classes)
# --ect k error correcting tournament
# --bfgs