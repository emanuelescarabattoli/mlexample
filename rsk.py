import os
import pandas as pd

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Here we define the path of our data file
# the file contains a list of ratings that
# user gives to given items
PATH = os.path.dirname(os.path.realpath(__file__))
RATINGS_PATH = PATH + "/data/movies/u.data"

# We read all the  ratings from the data file
all_ratings = pd.read_csv(
    RATINGS_PATH, sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
)

# Getting the count of users and items
# by using the max value found in the ids
max_user_id = all_ratings["user_id"].max()
max_item_id = all_ratings["item_id"].max()

# We split the dataset in two part:
# one to train the model
# and one to test it
train_ratings, test_ratings = train_test_split(
    all_ratings, test_size=0.2, random_state=0
)

# Getting the list of user ids, items ids and ratings
# used to train the model
user_id_train = train_ratings["user_id"]
item_id_train = train_ratings["item_id"]
rating_train = train_ratings["rating"]

# Getting the list of user ids, items ids and ratings
# used to test the model
user_id_test = test_ratings["user_id"]
item_id_test = test_ratings["item_id"]
rating_test = test_ratings["rating"]

# Here we define the two input layer for the model:
# starting from a list of user and items
# we want to predict the rating that an user
# will give to a given item
user_input_layer = keras.layers.Input(shape=[1], name="user_input")
item_input_layer = keras.layers.Input(shape=[1], name="item_input")

# The embedding size is defined here
# TODO: try to understand the meaning of this value
embedding_size = 11

# After creating two input layer, we use the output of
# these layers as the input of this two embedding layers
# TODO: it is not clear what these layers does
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

# Using the embedding layer as input of flatten layers
user_flatten_layer = keras.layers.Flatten()(user_embedding_layer)
item_flatten_layer = keras.layers.Flatten()(item_embedding_layer)

# Here we concatenate the two layers
# TODO: need to understand why
input_vectors = keras.layers.concatenate([user_flatten_layer, item_flatten_layer])

# Here we dropout some values to avoid overfitting the model
input_vectors = keras.layers.Dropout(0.5)(input_vectors)

# From the result of previous layer we create the input
# for the final layer, the output layer
x = keras.layers.Dense(128, activation="relu")(input_vectors)

# We want to have a single output from out model,
# so we use this layer
y = keras.layers.Dense(1)(x)

# Compiling the model starting form train data
model = keras.models.Model(inputs=[user_input_layer, item_input_layer], outputs=[y])
model.compile(optimizer="adam", loss="mae")

# Fitting the model to train it
model.fit(
    [user_id_train, item_id_train],
    rating_train,
    batch_size=64,
    epochs=20,
    validation_split=0.1,
    shuffle=True,
    verbose=2,
)

# Getting predictions from the model
train_predictions = model.predict([user_id_train, item_id_train]).squeeze()
test_predictions = model.predict([user_id_test, item_id_test]).squeeze()

# Printing the mean absolute error to evaluate the accuracy of the model
print(mean_absolute_error(train_predictions, rating_train))
print(mean_absolute_error(test_predictions, rating_test))
