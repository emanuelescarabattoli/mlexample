import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PATH = os.path.dirname(os.path.realpath(__file__))
RATINGS_PATH = PATH + "/data/movies/ratings.csv"

ratings = pd.read_csv(RATINGS_PATH)

group_users = ratings.groupby('userId')['rating'].count()
top_users = group_users.sort_values(ascending=False)[:15]

group_ratings = ratings.groupby('movieId')['rating'].count()
top_movies = group_ratings.sort_values(ascending=False)[:15]

top_ratings = ratings.join(top_users, rsuffix='_r', how='inner', on='userId')
top_ratings = top_ratings.join(top_movies, rsuffix='_r', how='inner', on='movieId')

# print(pd.crosstab(top_ratings.userId, top_ratings.movieId, top_ratings.rating, aggfunc=np.sum))

user_encoder = LabelEncoder()
ratings['user'] = user_encoder.fit_transform(ratings['userId'].values)
number_users = ratings['user'].nunique()

item_encoder = LabelEncoder()
ratings['movie'] = item_encoder.fit_transform(ratings['movieId'].values)
number_movies = ratings['movie'].nunique()

ratings['rating'] = ratings['rating'].values.astype(np.float32)
min_rating = min(ratings['rating'])
max_rating = max(ratings['rating'])

# print(number_users, number_movies, min_rating, max_rating)

x = ratings[['user', 'movie']].values
y = ratings['rating'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

number_factors = 50

x_train_array = [x_train[:, 0], x_train[:, 1]]
x_test_array = [x_test[:, 0], x_test[:, 1]]