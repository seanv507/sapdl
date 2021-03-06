import hashlib, uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold

from keras import models, layers, optimizers, regularizers
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
import os

u = str(uuid.uuid4())

def save_results(ident, model, hist, models, results):
    model_yaml = model.to_yaml()
    with open(ident + '_model.yaml', 'w') as f:
        f.write(model_yaml)
    res = pd.DataFrame(hist.history)
    res.to_csv(ident  + '_results.csv')
    res['run'] = ident
    results[ident] = res
    models[ident ] = model.to_yaml()
    return models, results


def plot_history(hi):
    history_dict = hi.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


dfs = ( pd.read_csv('offline_challenge/xtrain_obfuscated.txt', header=None, names=['text']),
        pd.read_csv('offline_challenge/ytrain.txt',header=None, names=['label']))
df = pd.concat(dfs,axis=1)
labels = pd.read_csv('labels.txt',header=None, names=['label','title'], sep=' ')
df =df.merge(labels, on='label')
df['len']=df.text.str.len()

# df.groupby('title')['len'].hist(normed=True)
# normed doesn't seem to work
count_vec = CountVectorizer(analyzer='char', ngram_range=(1,6))
text = count_vec.fit_transform(df.text)

cnts = text.sum(axis=0)
z = pd.Series(np.asarray(cnts).squeeze(), index = count_vec.get_feature_names())

vc = z.value_counts().sort_index(ascending=False)

# identify features
#
poly_feat = PolynomialFeatures(2)
poly_text = poly_feat.fit_transform(text)

categorical_labels = to_categorical(df.label, num_classes=None)

text_train, text_test, categorical_labels_train, categorical_labels_test = (
        train_test_split(text, categorical_labels, random_state=42))


res_all = {}
models_all = {}


n_output = len(labels)
model = models.Sequential()
model.add(layers.Dense(100, activation='relu', input_shape = (text.shape[1],)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(n_output,
                       kernel_regularizer=regularizers.l2(0.001),
                       activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='categorical_crossentropy')
hist = model.fit(text_train, categorical_labels_train, epochs=20,
                 batch_size=512, validation_data=(text_test, categorical_labels_test))
u = str(uuid.uuid4())

models_all, res_all= save_results(u, model, hist, models_all, res_all)

#hist_100_dr 2 layers of 100, dropout of 0.2, l2=0.0001
model_lin = models.Sequential()
model_lin.add(layers.Dense(n_output, activation='softmax',
                           kernel_regularizer=regularizers.l1(0.0001),
                           input_shape = (text.shape[1],)))
model_lin.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='categorical_crossentropy')
hist_lin = model_lin.fit(text_train, categorical_labels_train, epochs=20,
                 batch_size=512, validation_data=(text_test, categorical_labels_test))
u_lin = str(uuid.uuid4())
# how to ensure write only once? use dictionaries and pass identifier
models_all, res_all = save_results(u_lin, model_lin, hist_lin, models_all, res_all)





# regularisation l1/ l2
# embedding
# drop out
# try single value (replicate vowpal wabbit)

# tokenize decide


#vw_data = (df.label + 1).astype(str).str.cat(df.text, sep=' | ').sample(frac=1)
#vw_data.to_csv('offline_challenge/vw_data.txt', header=None, index=False)

# vw --passes=15 --cache --ngram=10 --skips=3 --early_terminate=15 --l1=0.0001 --loss_function=logistic --oaa 12 vw_data.txt -f vw_data.model
tokenizer = Tokenizer(char_level=True)

tokenizer.fit_on_texts(df.text.tolist())
sequences =  tokenizer.texts_to_sequences(df.text.tolist())
word_index = tokenizer.word_index
print('Found {} unique tokens.'.format(len(word_index)))
text

plot_history(hist)

plot_history(hist_lin)

# questions are there certain pairs that are more likely

with open('gs1') as f:
    gs1=pickle.load(f)

#./vw -d train.txt -f lg.vw --loss_function logistic
# cache
#./vw -d train.txt -c --passes 10


# test and save predictions
#./vw -d test.txt -t -i predictor.vw -p predictions.txt

# multiclass Label in {1,2,...,k}
# --oaa k (one against all k classes)
# --ect k error correcting tournament
# --bfgs