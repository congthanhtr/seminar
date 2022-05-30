#!/usr/bin/env python
# coding: utf-8

# In[23]:


#!/usr/bin/python

#-----------------------------------------------------------------------
# twitter-retweets
#  - print who has retweeted tweets from a given user's timeline
#-----------------------------------------------------------------------

from twitter import *
from pymongo import MongoClient
import json
import time

client = MongoClient("mongodb://localhost:27017/")
db = client["twitter"]
print(client.list_database_names())


# In[24]:


#-----------------------------------------------------------------------
# load our API credentials 
#-----------------------------------------------------------------------
config = {"access_key": "1524010849307820033-lOHNswzdGsWMqsS9MfPs32WJKsbyOe", "access_secret": "XeVjRuKu9v1tuZAhpJiD57EVXVGrr18r6S46wmncmWmZC", "consumer_key": "MBMOo6lLqogI6KUe2F7V56wff", "consumer_secret": "U9WIjWXe1TKprUsqttgvn1W8RHxYcDjnOrydEzHeWPzAlPIsiL"}
# execfile("config.py", config)


# In[25]:


def pp_json(json_thing, sort=True, indents=4):
    if type(json_thing) is str:
        print(json.dumps(json.loads(json_thing), sort_keys=sort, indent=indents))
    else:
        print(json.dumps(json_thing, sort_keys=sort, indent=indents))
    return None


# In[26]:


#-----------------------------------------------------------------------
# create twitter API object
#-----------------------------------------------------------------------
twitter = Twitter(auth = OAuth(config["access_key"], config["access_secret"], config["consumer_key"], config["consumer_secret"]))
# In[27]:


#-----------------------------------------------------------------------
# loop through each of my statuses, and print its content
#-----------------------------------------------------------------------
def get_posts(user):
    # print (db.posts.find({"user.screen_name": user}).count())
    prev_id = 0
    for status in db.posts.find({"user.screen_name": user}).sort([("id", 1)]).limit(1):
        prev_id = status["id"]

    while True:
        try:
            if prev_id != 0:
                results = twitter.statuses.user_timeline(screen_name=user, max_id = prev_id, count=200)
            elif last_id != 0:
                results = twitter.statuses.user_timeline(screen_name=user, since_id = last_id, count=200)
            else:
                results = twitter.statuses.user_timeline(screen_name=user, count=200)
            for status in results:
                #print "@%s %s \n%s" % (user, status["id"], status["text"])
                db.posts.insert_one(status)
            print (db.posts.find({"user.screen_name": user}).count())
            prev_id = results[-1]["id"]
        except:
            print ("sleeping...")
            pp_json(twitter.application.rate_limit_status()["resources"]["statuses"]["/statuses/user_timeline"])
            time.sleep(60*15)
# In[28]:


#-----------------------------------------------------------------------
# loop through each of user's statuses, and get retweeter list 
#-----------------------------------------------------------------------
def get_retweeter_list_from_posts(user):
    print (db.posts.find({"user.screen_name": user}).count())
    users = {}
    i = 0
    for status in db.posts.find({"user.screen_name": user}, {"id":1, "text":1, "created_at":1}, no_cursor_timeout=True):
        if db.retweeter_list.find_one({"tweet_id": status["id"]}):
            continue
        else:
            #print "@%s %s %s " % (user, status["text"], status["created_at"]) 
            #print status["id"]
            if i%100 == 0:
                pp_json(twitter.application.rate_limit_status()["resources"]["statuses"]["/statuses/retweeters/ids"])
                print (db.retweeter_list.count())
            i += 1
            try:
                retweets = twitter.statuses.retweeters.ids(_id=status["id"])
                db.retweeter_list.update({"tweet_id": status["id"]}, {'$set': {"retweeters": retweets["ids"]}}, upsert=True)
            except:
                print ("sleeping...")
                pp_json(twitter.application.rate_limit_status()["resources"]["statuses"]["/statuses/retweeters/ids"])
                time.sleep(60*15)
# In[29]:


#-----------------------------------------------------------------------
# loop through retweeter list, and get the users infomation
#-----------------------------------------------------------------------
def get_users_from_retweeter_list():
    i = 0
    count = 0
    for retweeter_list in db.retweeter_list.find({}, {"tweet_id":1, "retweeters":1}, no_cursor_timeout=True):
        if db.users.find_one({"tweet_id": retweeter_list["tweet_id"] }):
            continue
        else:
            
            if i%100 == 0:
                pp_json(twitter.application.rate_limit_status()["resources"]["users"]["/users/lookup"])
            i += 1
            try:
                if len(retweeter_list["retweeters"]) > 0:
                    retweeters = twitter.users.lookup(user_id=retweeter_list["retweeters"])
                    count += 1
                    for i in range(len(retweeters)):
                        print ("@%s %s " % (retweeters[i]["id"], retweeters[i]["name"]))
                        db.users.insert_one({"tweet_id": retweeter_list["tweet_id"], "user": retweeters[i]})
                    print ("number of retweeters: ", retweeter_list["tweet_id"], len(retweeter_list["retweeters"]))
                if count == 900:
                    print ("sleeping...")
                    count = 0
                    pp_json(twitter.application.rate_limit_status()["resources"]["users"]["/users/lookup"])
                    time.sleep(60*15)
            except:
                continue

# In[30]:


counts = {}
for retweeter_list in db.retweeter_list.find({}, {"tweet_id":1, "retweeters":1}, no_cursor_timeout=True):
    name = db.posts.find_one({"id": retweeter_list["tweet_id"]})["user"]["screen_name"]
    db.retweeter_list.update_one({"tweet_id": retweeter_list["tweet_id"]}, {'$set': {"source": name}}, upsert=True)


# In[15]:


print (counts)


# In[ ]:


for user in db.users.find({}, {"tweet_id":1}, no_cursor_timeout=True):
    name = db.posts.find_one({"id": user["tweet_id"]})["user"]["screen_name"]
    db.users.update_one({"tweet_id": user["tweet_id"]}, {'$set': {"source": name}}, upsert=True)


# In[1]:


import newspaper
from newspaper import Article


# In[2]:


def get_article(url):
    a = Article(url, language='en')
    a.download()
    a.parse()
    return a


# In[3]:


def article_to_document(article, idx):
    document = {"top_image":article.top_image, "text": article.text, "title": article.title, "id": idx, "authors": article.authors}
    document["images"] = article.images
    document["movies"] = article.movies
    article.nlp()
    document["summary"] = article.summary
    document["keywords"] = article.keywords
    return document


# In[ ]:



