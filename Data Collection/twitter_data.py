from twikit import Client, TooManyRequests
from datetime import datetime
import time
import csv
import os
from random import randint
import asyncio
from configparser import ConfigParser

full_path = os.path.join(os.getcwd(), 'Data Collection', 'twitter_config.ini')


# get login information from the twitter_config.ini file
config = ConfigParser()
config.read(full_path)
username = config['X']['username']
email = config['X']['email']
password = config['X']['password']

# authenticate to X.com using the login credentials
client = Client(language='en')
async def login():
    await client.login(auth_info_1=username, auth_info_2=email, password=password)
    client.save_cookies('cookies.json')

asyncio.run(login())
client.load_cookies('cookies.json')


# create csv file to save tweets
with open('twitter_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Tweet Number', 'Username', 'Text', 'Created At', 'Retweets', 'Likes'])
    

# query tweets
MINIMUM_TWEETS = 10000
QUERY=''
    
def get_tweets(tweets):
    if tweets is None:
        tweets = client.search_tweet(QUERY, product='Top')
    else:
        wait_time = randint(5, 10)
        time.sleep(wait_time)
        tweets = tweets.next()

    return tweets

tweet_number = 0
tweets = None
    
while tweet_number < MINIMUM_TWEETS:

    try:
        tweets = get_tweets(tweets)
    except TooManyRequests as e:
        rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
        print(f'{datetime.now()} - Rate limit reached. Reset time: {rate_limit_reset}')
        wait_time = rate_limit_reset - datetime.now()
        time.sleep(wait_time.total_seconds())
        continue

    if not tweets:
        break

    for tweet in tweets:
        tweet_number += 1
        tweet_data = [tweet_number, tweet.user.name, tweet.text, tweet.created_at, tweet.retweet_count, tweet.favorite_count]
        
        with open('twitter_data.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(tweet_data)



