import tweepy
import pandas as pd
from textblob import TextBlob
import os
consumer_key = "c3VJToNvMBggAhAaJGvOhR9bn"
consumer_secret = "e0Ga4MpNA3xmYsub1qyxcWHV46mZhvSur5Ck5rJUEb7NWJyjzs"
access_token = "1739275641638342656-r4zKccSiXttMbDMRI75i4PQaWsBDEA"
access_token_secret = "5LSDxdNE1iCdADSz52JtjDPr5oPk0YRKsPURwYY0ktnnN"
# Настройка API ключей и токенов
# api_key = 'YOUR_API_KEY'
# api_secret_key = 'YOUR_API_SECRET_KEY'
# access_token = 'YOUR_ACCESS_TOKEN'
# access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'
client_id = "cG5ERTRJd05YdjdCSC1jeTZtTmw6MTpjaQ"
client_secret = "lGtiRWqnr4ugCK1-24fn3-bC3Eduav506GBC-wsF8D8A2NUp1r"
# Аутентификация с помощью Tweepy
# auth = tweepy.OAuthHandler(api_key, api_secret_key)
# auth.set_access_token(access_token, access_token_secret)
# api = tweepy.API(auth)


# auth = tweepy.OAuth2BearerHandler("AAAAAAAAAAAAAAAAAAAAAMTjtwEAAAAA4SuI3E%2FNAcnrbQD9b6WZZyLHCxw%3DYTJ0En5HX6meLUAwbhBUdKNT1OS7ehUaAGy1KKq1uXQGeEMNmo")
# api = tweepy.API(auth)
bearer_token = "Bearer AAAAAAAAAAAAAAAAAAAAAMTjtwEAAAAA4SuI3E%2FNAcnrbQD9b6WZZyLHCxw%3DYTJ0En5HX6meLUAwbhBUdKNT1OS7ehUaAGy1KKq1uXQGeEMNmo"
# client = tweepy.Client(bearer_token=bearer_token)
auth = tweepy.OAuth1UserHandler(
    consumer_key, consumer_secret, access_token, access_token_secret
)
api = tweepy.API(auth)
# Запрос для получения последних твитов из вашей ленты
public_tweets = api.search_tweets(q="bitcoin")

# Вывод полученных твитов
for tweet in public_tweets:
    print(tweet.text)
# Функция для получения твитов и анализа тональности
def fetch_tweets(keyword, count=100):
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang='en').items(count)
    data = []

    for tweet in tweets:
        text = tweet.text
        username = tweet.user.screen_name
        created_at = tweet.created_at

        # Анализ тональности с TextBlob
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity

        data.append([created_at, username, text, polarity, subjectivity])

    return pd.DataFrame(data, columns=['DateTime', 'Username', 'Text', 'Polarity', 'Subjectivity'])


# Сбор твитов по ключевому слову 'Bitcoin'
# df = fetch_tweets('Bitcoin', count=1000)
#
# # Сохранение данных в CSV
# df.to_csv('bitcoin_tweets.csv', index=False)
# print(df.head())