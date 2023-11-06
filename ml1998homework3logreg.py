import pandas as pd
from sklearn.model_selection import train_test_split

sentweet = pd.read_csv('./data/tweet_data_2.csv')
print(sentweet.shape)

sentweet['tweet'] = sentweet.apply(lambda row: str(row['tweet']).lower(), axis=1)
from string import punctuation


def remove_punctuations(string):
    return ''.join(c for c in string if c not in punctuation)


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))


def remove_stopwords(string):
    tokenized = word_tokenize(string)
    filtered_sentence = [word for word in tokenized if not word in stop_words]
    return ' '.join(c for c in filtered_sentence)


sentweet['tweet'] = sentweet.apply(lambda row: remove_stopwords(row['tweet']), axis=1)


def convert(integer):
    if integer > 0:
        return 'Positive'
    elif integer == 0:
        return 'Neutral'
    else:
        return 'Negative'


sentweet['polarity'] = sentweet.apply(lambda row: convert(row['polarity']), axis=1)

X = sentweet['tweet']
y = sentweet['polarity']

one_hot_encoded_label = pd.get_dummies(y)
print(one_hot_encoded_label.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(min_df=2, ngram_range=(1, 1))
X_train = vect.fit(X_train).transform(X_train)
X_test = vect.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

c_val = [0.75, 1, 2, 3, 4, 5, 10]

for c in c_val:
    logreg = LogisticRegression(C=c)
    logreg.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_test, logreg.predict(X_test))))