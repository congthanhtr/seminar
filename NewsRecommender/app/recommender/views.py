#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from flask import Flask
from flask import render_template
from recommender import app
from pymongo import MongoClient
import json,pickle
import os
from recommender import request as rq
from flask import request, url_for, render_template
client = MongoClient()
db = client.twitter

APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top

@app.route('/')
def home():
    return render_template('my_home.html')

@app.route('/news/')
def news():
    return render_template('news.html')

@app.route('/how/')
def how():
    return render_template('how.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/get_news/', methods=["GET", "POST"])
def get_news():
    if request.method == "POST":
        query = request.form.get('query')
        print("running in get_news with " + query)
        data = rq.get_recommendations(query,5)
    return render_template('news_recommend.html', data=data)
  
if __name__ == "__main__":
    app = Flask(__name__)
    app.run(debug=True)
