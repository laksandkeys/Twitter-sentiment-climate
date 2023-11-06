import pandas as pd
import numpy as py

sentweet = pd.read_csv('./data/tweet_data_2.csv')

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
doc_term_matrix = count_vect.fit_transform(sentweet['tweet'].values.astype('U'))

from sklearn.decomposition import LatentDirichletAllocation

LDA = LatentDirichletAllocation(n_components=5, random_state=42)
LDA.fit(doc_term_matrix)

import random

for i in range(10):
    random_id = random.randint(0,len(count_vect.get_feature_names()))
    print(count_vect.get_feature_names()[random_id])

first_topic = LDA.components_[0]
top_topic_words = first_topic.argsort()[-10:]

for i in top_topic_words:
    print(count_vect.get_feature_names()[i])

for i,topic in enumerate(LDA.components_):
    print(f'Top 10 words for topic #{i}:')
    print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')

topic_values = LDA.transform(doc_term_matrix)
print(topic_values.shape)

sentweet['Topic'] = topic_values.argmax(axis=1)
print(sentweet.head())

sentweet.to_csv('/Users/user/Desktop/tweet_data_3.csv')

