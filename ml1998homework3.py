import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tweepy as tw

import re

from textblob import TextBlob

import warnings

warnings.filterwarnings("ignore")

sns.set(font_scale=1.5)
sns.set_style("whitegrid")

consumer_key = 'LrHT2tqUgSzXMVwyUqlh9FS2C'
consumer_secret = '6vOZPHdThL6VMQ0sdgFGrSCJnT3v4JIvGh5Stp6UkQcC5LwI0a'
access_token = '1312120872505937921-wVeVmamBcEWqNmJVHLMhe63rjBsPJI'
access_token_secret = 'h1ApNtdKW8iPvefs69rJEcA4AoOxHcl163oEdV2pNPyyZ'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)


def remove_url(txt):
    """Replace URLs found in a text string with nothing
    (i.e. it will remove the URL from the string).

    Parameters
    ----------
    txt : string
        A text string that you want to parse and remove urls.

    Returns
    -------
    The same txt string with url's removed.
    """

    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())


search_term = "#climate+change -filter:retweets"

tweets = tw.Cursor(api.search,
                   q=search_term,
                   lang="en",
                   since='2018-11-01').items(1000)

# Remove URLs
tweets_no_urls = [remove_url(tweet.text) for tweet in tweets]

sentiment_objects = [TextBlob(tweet) for tweet in tweets_no_urls]

sentiment_objects[0].polarity, sentiment_objects[0]
# Create list of polarity values and tweet text
sentiment_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in sentiment_objects]

sentiment_df = pd.DataFrame(sentiment_values, columns=["polarity", "tweet"])

print(sentiment_df.head())

fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram of the polarity values
sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
                  ax=ax,
                  color="green")

plt.title("Sentiments from Tweets on Climate Change")
plt.show()

sentiment_df = sentiment_df[sentiment_df.polarity != 0]
fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram with break at zero
sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1],
                  ax=ax,
                  color="green")

plt.title("Sentiments from Tweets on Climate Change")
plt.show()

sentiment_df.to_csv('/Users/user/Desktop/tweet_data_2.csv')