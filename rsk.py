import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras


PATH = os.path.dirname(os.path.realpath(__file__))
RATINGS_PATH = PATH + "/data/movies/u.data"

all_ratings = pd.read_csv(
    RATINGS_PATH, sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
)

max_user_id = all_ratings["user_id"].max()
max_item_id = all_ratings["item_id"].max()

train_ratings, test_ratings = train_test_split(
    all_ratings, test_size=0.2, random_state=0
)

user_id_train = train_ratings["user_id"]
item_id_train = train_ratings["item_id"]
rating_train = train_ratings["rating"]


user_id_test = test_ratings["user_id"]
item_id_test = test_ratings["item_id"]
rating_test = test_ratings["rating"]

user_input_layer = keras.layers.Input(shape=[1], name="user_input")
item_input_layer = keras.layers.Input(shape=[1], name="item_input")

embedding_size = 11

user_embedding_layer = keras.layers.Embedding(
    output_dim=embedding_size,
    input_dim=max_user_id + 1,
    input_length=1,
    name="user_embedding",
)(user_input_layer)
item_embedding_layer = keras.layers.Embedding(
    output_dim=embedding_size,
    input_dim=max_item_id + 1,
    input_length=1,
    name="item_embedding",
)(item_input_layer)

user_flatten_layer = keras.layers.Flatten()(user_embedding_layer)
item_flatten_layer = keras.layers.Flatten()(item_embedding_layer)

input_vectors = keras.layers.concatenate([user_flatten_layer, item_flatten_layer])

input_vectors = keras.layers.Dropout(0.5)(input_vectors)

x = keras.layers.Dense(128, activation="relu")(input_vectors)

y = keras.layers.Dense(1)(x)

model = keras.models.Model(inputs=[user_input_layer, item_input_layer], outputs=[y])
model.compile(optimizer="adam", loss="mae")

history = model.fit(
    [user_id_train, item_id_train],
    rating_train,
    batch_size=64,
    epochs=20,
    validation_split=0.1,
    shuffle=True,
    verbose=2,
)

initial_train_predictions = model.predict([user_id_train, item_id_train]).squeeze()
