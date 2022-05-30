import tweepy
import pandas as pd
import time

consumer_key = "MBMOo6lLqogI6KUe2F7V56wff"
consumer_secret = "U9WIjWXe1TKprUsqttgvn1W8RHxYcDjnOrydEzHeWPzAlPIsiL"
access_token = "1524010849307820033-lOHNswzdGsWMqsS9MfPs32WJKsbyOe"
access_token_secret = "XeVjRuKu9v1tuZAhpJiD57EVXVGrr18r6S46wmncmWmZC"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

username = "congthanhtr151"
count = 1

try:     
    # Creation of query method using parameters
    tweets = tweepy.Cursor(api.user_timeline, id=username).items(count)
    
    # Pulling information from tweets iterable object
    tweets_list = [[tweet.created_at, tweet.id, tweet.text] for tweet in tweets]
    
    # Creation of dataframe from tweets list
    # Add or remove columns as you remove tweet information
    tweets_df = pd.DataFrame(tweets_list)
    print(tweets_df)
 
except BaseException as e:
      print('failed on_status,',str(e))
      time.sleep(3)