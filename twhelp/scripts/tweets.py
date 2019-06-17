import tweepy
import sys
import re
import csv
import pandas
import random
consumer_key = 'HXU2TGD1jYykI0nZttUr3gqna'
consumer_secret = 'PRlOQewKL3yjqq04VfSY6i12JF5bzll9TjI4hL8VkhYyLajFJr'
access_key = '1127147275069816832-FdAJXupVKcfZfQaAc1s0Tdm3y8PFeV'
access_secret = 'oy2EnfmCezkibJObHVEOgbh3KYolGjBNTyVednFfQhoYO'


def to_utf(value):
  return value.encode('utf-8', 'ignore')


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

try:
  api = tweepy.API(auth, wait_on_rate_limit=True)
except tweepy.error.TweepError:
  pass
  
  
random.seed(42)
fields = ['Text', 'Date', 'Likes', 'Retweets']
rows = []
n = 200
if sys.argv[2] is not None:
    n = int(to_utf(sys.argv[2]))

with open("../data/tweets_data.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile, delimiter='\t')
    csvwriter.writerow(fields)
    for stuff in tweepy.Cursor(
        api.user_timeline, to_utf(sys.argv[1]),
        tweet_mode="extended", exclude_replies=True,
        include_rts = False).items(n):

        csvwriter.writerow([
            re.sub(r'http\S+', '', stuff.full_text), stuff.created_at,
            stuff.favorite_count,
            stuff.retweet_count
        ])


df = pandas.read_csv('../data/tweets_data.csv', delimiter='\t')
print(df)

