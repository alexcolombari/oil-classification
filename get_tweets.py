import sys
import tweepy
from textblob import TextBlob

# Consume
CONSUMER_KEY = '8o4sWzRw3s3xjVr7AMIugz1lL'
CONSUMER_SECRET = '450l4HpzQGuFGUw7aEnL7mGJh7tz45bUAbJyrhR17PxOKDxAgc'
# Access:
ACCESS_TOKEN = '902900523241754625-EJgcCZ5ftQ4cZ9JsYCjWtyZjeWhftOB'
ACCESS_TOKEN_SECRET = 'WFcU21JoxHHc5npA8W8ZCbhSBnC0WCmoJQEzlzC86BZ0v'

def connect_twitter_api():
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)

    return api

def buildTestSet(search_keyword, COUNT):
    api = connect_twitter_api()
    user = api.get_user()
    try:
        tweets_fetched = api.search(search_keyword, count = COUNT)
        print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + search_keyword + "\n")
        for tweet in tweets_fetched:
            analysis = TextBlob(tweet.text)
            if analysis.sentiment[0] == 0:
                print("\n", user + tweet.text)
                #analysis = TextBlob(tweet.text)
                print(analysis.sentiment)
            '''
            if analysis.sentiment[0] > 0:
                print("Positive")
            elif analysis.sentiment[0] < 0:
                print('Negative')
            else:
                print('Neutral')
            '''
    except:
        print('Unfortunately, something went wrong...')
        return None

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('[INFO] Uso: python get_tweets.py palavra-chave quantidade')
        sys.exit(1)

    search_term = str(sys.argv[1])
    COUNT = int(sys.argv[2])
    testDataSet = buildTestSet(search_term, COUNT)