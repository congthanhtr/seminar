import matplotlib.pyplot as plt
#%pip install "gensim" "spacy" "pyLDAvis" 
import gensim
import numpy as np
import spacy
import pandas as pd
import re
from gensim import similarities
import nltk
import pickle
nltk.download('stopwords')

from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from spacy.lang.en.stop_words import STOP_WORDS
import pyLDAvis.gensim_models
#Import nltk stopwords and add custom stopwords that are likely to appear in news articles.
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(["mrs","ms","say","he","mr","she","they","company"])

import os, re, operator, warnings
warnings.filterwarnings('ignore')

# load file csv to dataframe
# df=pd.read_csv("NewsArticles.csv", encoding='unicode_escape',index_col=0)
# #drop all the unnamed columns
# df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
with open('dataframe.txt', 'rb') as f:
    df = pickle.load(f)

data = df['text'].values.tolist()
data1 = df['title'].values.tolist()

# before loading the language you have to download it first. Go to your command prompt and execute this statement and 
# restart the kernel:
# python -m spacy download en_core_web_sm
# load spacy model
import en_core_web_sm
nlp = en_core_web_sm.load()

#removing punctuations and others characters
def preprocess(string):
    return re.sub('[^\w_\s-]', ' ',str(string))

data = list(map(preprocess,data))    

# get lemma doc
#data cleaning and lemmatization
lemma_doc = []
print("lemmaing...")
# for datum in data:
#     sent = nlp(str(datum).lower())
#     text = []
#     for w in sent:
#         if not w.is_stop and not w.is_punct and not w.like_num and str(w) not in stop_words and (len(str(w)) > 4):
#             #adding the lematized version of the words
#             text.append(w.lemma_)
#     lemma_doc.append(text)

with open('lemma.txt', 'rb') as f:
    lemma_doc = pickle.load(f)

# word2id
word2id = corpora.Dictionary(lemma_doc)

# Creates bag of words and a corpus
documents = lemma_doc
corpus = [word2id.doc2bow(doc) for doc in documents]

#https://towardsdatascience.com/lets-build-an-article-recommender-using-lda-f22d71b7143e
def get_similarity(lda, query_vector):
    index = similarities.MatrixSimilarity(lda[corpus])
    
    sims = index[query_vector]
    return sims

#https://github.com/RaRe-Technologies/gensim/issues/2644
# its taking me 3 days for this fucking issue!!!!
query="Donald Trump"
words = word2id.doc2bow(query.split())

# lda model
lda_model = LdaModel(corpus=corpus, id2word=word2id, num_topics=5, random_state=42, update_every=1, chunksize=100, 
                     passes=10, alpha='auto')

print("Top words identified: ")
for word in words:
    print("{} {}".format(word[0], word2id[word[0]]))


query_vector = lda_model[words]
print(query_vector)

sims = get_similarity(lda_model, query_vector)

sims = sorted(enumerate(sims), key=lambda item: -item[1])

idx = 0
pids = []
result = 10
article_ids = df['article_source_link'].values.tolist()

print("\nCheck out the links below:")
while result > 0:
    pageid = article_ids[sims[idx][0]]
    if pageid not in pids:
        pids.append(pageid)
        print("{}".format(pageid))
        result -= 1
    idx += 1

