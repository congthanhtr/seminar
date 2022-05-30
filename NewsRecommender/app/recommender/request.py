#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from tkinter.messagebox import NO
import numpy as np
import re
from time import sleep
import operator
import pickle
import random
from flask import request

from pymongo import MongoClient
from wordcloud import WordCloud, STOPWORDS

client = MongoClient()
db = client.twitter


def get_articles_for_group(size=10, group=-1):
    articles = []
    article_list = db.group_articles.find_one({"id": group})["articles"]
    i = 0
    titles = set()
    for article_id in article_list:
        if i == size*2:
            break
        article = db.rec_articles.find_one({"id": article_id})
        if article and len(article["summary"]) > 100 and article["title"] not in titles:
            entry = {}
            titles.add(article["title"])
            entry["title"] = article["title"]
            entry["url"] = article["url"]["url"]
            entry["source"] = article["source"]["name"]
            entry["text"] = article["text"]
            entry["summary"] = article["summary"]
            entry["keywords"] = article["keywords"]
            entry["authors"] = article["authors"]
            entry["images"] = article["images"]
            entry["top_image"] = article["top_image"]
            entry["movies"] = article["movies"]
            articles.append(entry)
            i += 1
    return random.sample(articles, size)


def majority_vote(labels):
    votes = {}
    for i in labels:
        if i in votes:
            votes[i] += 1
        else:
            votes[i] = 1
    if len(votes) > 0:
        return max(votes.iteritems(), key=operator.itemgetter(1))[0]
    else:
        return 5


def predict_user_group(name, validate=False):
    user_active = db.active_users.find_one({"handle": name})
    user_not_active = db.user_merge.find_one({"handle": name})
    if not validate and user_active:
        return user_active["label"]
    elif user_not_active:
        labels = []
        for retweet in user_not_active["tweet_ids"]:
            article = db.articles.find_one({"id": retweet})
            if article and "label" in article:
                labels.append(article["label"])
        return majority_vote(labels)
    else:
        return -1


def save_articles(articles):
    pickle.dump(articles, open('./articles.pkl', 'wb'))


def to_full_vec(group_topics, num_topics):
    topic_vec = [0 for j in range(num_topics)]
    for prob in group_topics:
        topic_vec[prob[0]] = prob[1]
    return topic_vec


def get_word_frequency(num_topics, lda_topics, topic_vec):
    dict_words = {}
    for i in range(num_topics):
        for word in lda_topics[i].split(":")[1].split(" "):
            if word in dict_words:
                dict_words[word] += topic_vec[i]
            else:
                dict_words[word] = topic_vec[i]

    stopwords = set(STOPWORDS)
    stopwords.add("new")
    stopwords.add("time")
    stopwords.add("main")
    stopwords.add("continue")
    stopwords.add("please")
    stopwords.add("president")
    stopwords.add("advertisement")
    stopwords.add("first")
    for word in stopwords:
        if word in dict_words:
            dict_words.pop(word, None)
    return dict_words


def collect_words():
    num_topics = 20
    dict_words = {}
    tot_topic_vec = np.zeros(num_topics)
    for group in range(6):
        group_topics = pickle.load(open('./group_lda_bow_topics.pkl', 'rb'))
        lda_topics = pickle.load(open('./pub_lda_bow_topics.pkl', 'rb'))
        topic_vec = to_full_vec(group_topics[group], num_topics)
        tot_topic_vec += np.array(topic_vec)
        dict_words[group] = get_word_frequency(
            num_topics, lda_topics, topic_vec)

    tot_topic_vec /= 6.0
    dict_words[-1] = get_word_frequency(num_topics, lda_topics, tot_topic_vec)
    pickle.dump(dict_words, open('./word_frequency.pkl', 'wb'))


def get_words(group):
    return pickle.load(open('./word_frequency.pkl', 'rb'))[group]


# def get_recommendations(name):
#     group = predict_user_group(name)
#     articles = get_articles_for_group(size=10, group=group)
#     # save_articles(articles)
#     #articles = pickle.load(open('./articles.pkl', 'rb'))
#     words = get_words(group)
#     return {"articles": articles, "words": words, "group": group}

# start from here
from gensim import similarities
from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
import pickle
import nltk
# nltk.download('stopwords')
import spacy
import pandas as pd

# load file csv to dataframe
# df=pd.read_csv("NewsArticles.csv", encoding='unicode_escape',index_col=0)                                                              
# #drop all the unnamed columns
# df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

with open('dataframe.txt', 'rb') as f:
    df = pickle.load(f)
    f.close()
data = df['text'].values.tolist()
data1 = df['title'].values.tolist()

# python -m spacy download en_core_web_sm
# load spacy model
import en_core_web_sm
nlp = en_core_web_sm.load()

#removing punctuations and others characters
def preprocess(string):
    return re.sub('[^\w_\s-]', ' ',str(string))

data = list(map(preprocess,data))    


lemma_doc = []
with open('lemma.txt', 'rb') as f:
    lemma_doc = pickle.load(f)
    f.close()

word2id = corpora.Dictionary(lemma_doc)

documents = lemma_doc
corpus = [word2id.doc2bow(doc) for doc in documents]

def get_similarity(lda, query_vector):
    index = similarities.MatrixSimilarity(lda[corpus])
    
    sims = index[query_vector]
    return sims

import urllib.request 
import requests
from bs4 import BeautifulSoup

def get_contents(link):
    html = requests.get(link)
    htmlParse = BeautifulSoup(html.text, 'html.parser')
    contents = ""
    for para in htmlParse.find_all("p"):
        contents = (contents + para.get_text())
    return contents

def get_image(link):
    r = requests.get(link)
    html = r.text
    soup = BeautifulSoup(html, "html5lib")
    links = soup.find_all('div', {'class': 'image'})
    if links:
        return (links[0].find('img')['src'])
    else:
        return ""

def get_title(link):
    request = requests.get(link)
    soup = BeautifulSoup(request.text, 'html.parser')
    title = soup.find('title')
    if title is None:
        return ""
    return title.string

def get_recommendations(query, result):
    # group = predict_user_group(query)
    # articles = get_articles_for_group(size=10, group=group)
    # # save_articles(articles)
    # #articles = pickle.load(open('./articles.pkl', 'rb'))
    # words = get_words(group)
    # return {"articles": articles, "words": words, "group": group}

    print("QUERY: " + query)
    words = word2id.doc2bow(query.lower().split())
    
    with open('lda_model.pkl','rb') as f:
        lda_model = pickle.load(f)
        f.close()

    query_vector = lda_model[words]

    sims = get_similarity(lda_model, query_vector)
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    idx = 0
    pids = []
    # article_ids = df['article_source_link'].values.tolist()
    with open('article_ids.txt','rb') as f:
        article_ids = pickle.load(f)
        f.close()

    # while result > 0:
    #     pageid = article_ids[sims[idx][0]]
    #     if pageid not in pids:
    #         pids.append(pageid)
    #         result -= 1
    #     idx += 1
    # return pids
    dict = []

    while result > 0:
        pageid = article_ids[sims[idx][0]]
        if pageid not in pids:
            # dict['image'] = get_image(pageid)
            # dict['title'] = get_title(pageid)
            # dict['content'] = get_contents(pageid)
            tuple = []
            tuple.append(get_image(pageid))
            tuple.append(get_title(pageid))
            tuple.append(get_contents(pageid))
            tuple.append(pageid)
            dict.append(tuple)
            result-=1
        idx+=1
    return dict