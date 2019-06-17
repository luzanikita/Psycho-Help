import re
import csv
import sys
import json
import tweepy
import pandas
import random


with open('../data/twitter_keys.json') as src:
  keys = json.load(src)
  src.close()

auth = tweepy.OAuthHandler(keys['consumer_key'], keys['consumer_secret'])
auth.set_access_token(keys['access_key'], keys['access_secret'])

try:
  api = tweepy.API(auth, wait_on_rate_limit=True)
except tweepy.error.TweepError:
  pass
  
  
random.seed(42)
fields = ['Text', 'Date', 'Likes', 'Retweets']
rows = []
n = 200
if sys.argv[2] is not None:
    n = int(sys.argv[2])

with open("../data/tweets_data.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile, delimiter='\t')
    csvwriter.writerow(fields)
    for stuff in tweepy.Cursor(
        api.user_timeline, sys.argv[1],
        tweet_mode="extended", exclude_replies=True,
        include_rts = False).items(n):

        csvwriter.writerow([
            re.sub(r'http\S+', '', stuff.full_text), stuff.created_at,
            stuff.favorite_count,
            stuff.retweet_count
        ])


df = pandas.read_csv('../data/tweets_data.csv', delimiter='\t')
print(df)

