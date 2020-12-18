import numpy as np
import pandas as pd
import gc
import os
import json
from collections import Counter, defaultdict
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px
import re
import nltk
from gensim.models import word2vec
from sklearn.manifold import TSNE
year_pattern = r'([1-2][0-9]{3})'


def get_metadata():
    with open('/kaggle/input/arxiv/arxiv-metadata-oai-snapshot.json', 'r') as f:
        for line in f:
            yield line


def clean_sentence(val):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    regex = re.compile('([^\s\w]|_|\r|\n)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")

    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)

    sentence = " ".join(sentence)
    return sentence


STOP_WORDS = nltk.corpus.stopwords.words()

target_key = 'abstract'
abstracts = []

for idx, paper in enumerate(get_metadata()):
    if idx == 5000:
        break
    for k, v in json.loads(paper).items():
        if k == target_key:
            abstracts.append(clean_sentence(v).strip().split())

model = word2vec.Word2Vec(abstracts, size=100, window=20, min_count=200, workers=4)


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

tsne_plot(model)