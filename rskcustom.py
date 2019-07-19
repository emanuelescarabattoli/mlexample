import os
import sys
import random

import pandas as pd

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# Here we define the path of our data file
# the file contains a list of ratings that
# user gives to given items
PATH = os.path.dirname(os.path.realpath(__file__))
RATINGS_PATH = PATH + "/data/items/items.csv"
TEST_PATH = PATH + "/data/items/test.csv"
PREDICTIONS_PATH = PATH + "/data/items/predictions.csv"
MODEL_PATH = PATH + "/data/items/items"


def generate():
    """
    A function to generate a file containing
    simulated users, items and ratings
    """
    # A variable used to store the last user id
    user_id = 0

    # Generating first list of users
    users_first_group = []
    for index in range(1, 102):
        user_id += 1
        users_first_group.append(
            {
                "user_id": str(user_id),
                "user_description": "Samsung User #" + str(user_id),
            }
        )

    # Generating second list of users
    users_second_group = []
    for index in range(1, 102):
        user_id += 1
        users_second_group.append(
            {"user_id": str(user_id), "user_description": "Apple User #" + str(user_id)}
        )

    # Generating third list of users
    users_third_group = []
    for index in range(1, 102):
        user_id += 1
        users_third_group.append(
            {"user_id": str(user_id), "user_description": "Asus User #" + str(user_id)}
        )

    # Variable used to store the last item id
    item_id = 0

    # Genrating first list of items
    items_first_group = []
    for index in range(1, 102):
        item_id += 1
        items_first_group.append(
            {
                "item_id": str(item_id),
                "item_description": "Smartphone Samsung Model #" + str(item_id),
                "item_category_id": "1",
                "item_category_description": "Smartphone",
            }
        )

    # Genrating second list of items
    items_second_group = []
    for index in range(1, 12):
        item_id += 1
        items_second_group.append(
            {
                "item_id": str(item_id),
                "item_description": "Smartphone Apple Model #" + str(item_id),
                "item_category_id": "1",
                "item_category_description": "Smartphone",
            }
        )

    # Genrating third list of items
    items_third_group = []
    for index in range(1, 202):
        item_id += 1
        items_third_group.append(
            {
                "item_id": str(item_id),
                "item_description": "Smartphone Asus Model #" + str(item_id),
                "item_category_id": "1",
                "item_category_description": "Smartphone",
            }
        )

    # Genrating fourth list of items
    items_fourth_group = []
    for index in range(1, 52):
        item_id += 1
        items_fourth_group.append(
            {
                "item_id": str(item_id),
                "item_description": "Smartphone Charger For Android Model #"
                + str(item_id),
                "item_category_id": "2",
                "item_category_description": "Smartphone Charger",
            }
        )

    # Genrating fifth list of items
    items_fifth_group = []
    for index in range(1, 22):
        item_id += 1
        items_fifth_group.append(
            {
                "item_id": str(item_id),
                "item_description": "Smartphone Charger For Apple Model #"
                + str(item_id),
                "item_category_id": "2",
                "item_category_description": "Smartphone Charger",
            }
        )

    # Genrating sixth list of items
    items_sixth_group = []
    for index in range(1, 52):
        item_id += 1
        items_sixth_group.append(
            {
                "item_id": str(item_id),
                "item_description": "Smartphone Cover For Asus Model #" + str(item_id),
                "item_category_id": "3",
                "item_category_description": "Smartphone Cover",
            }
        )

    # Here we will store ratings for different items
    # made by different users
    ratings = []

    # Generating rating based on simulated user preferences
    # for the first group
    for user in users_first_group:
        for item in items_first_group:
            ratings.append(
                {
                    "user_id": str(user["user_id"]),
                    "user_description": user["user_description"],
                    "item_id": str(item["item_id"]),
                    "item_description": item["item_description"],
                    "item_category_id": item["item_category_id"],
                    "item_category_description": item["item_category_description"],
                    "rating_value": str(random.randint(3, 5)),
                }
            )
        for item in items_second_group:
            ratings.append(
                {
                    "user_id": str(user["user_id"]),
                    "user_description": user["user_description"],
                    "item_id": str(item["item_id"]),
                    "item_description": item["item_description"],
                    "item_category_id": item["item_category_id"],
                    "item_category_description": item["item_category_description"],
                    "rating_value": str(random.randint(0, 2)),
                }
            )
        for item in items_third_group:
            ratings.append(
                {
                    "user_id": str(user["user_id"]),
                    "user_description": user["user_description"],
                    "item_id": str(item["item_id"]),
                    "item_description": item["item_description"],
                    "item_category_id": item["item_category_id"],
                    "item_category_description": item["item_category_description"],
                    "rating_value": str(random.randint(1, 3)),
                }
            )
        for item in items_fourth_group:
            ratings.append(
                {
                    "user_id": str(user["user_id"]),
                    "user_description": user["user_description"],
                    "item_id": str(item["item_id"]),
                    "item_description": item["item_description"],
                    "item_category_id": item["item_category_id"],
                    "item_category_description": item["item_category_description"],
                    "rating_value": str(random.randint(3, 5)),
                }
            )
        for item in items_fifth_group:
            ratings.append(
                {
                    "user_id": str(user["user_id"]),
                    "user_description": user["user_description"],
                    "item_id": str(item["item_id"]),
                    "item_description": item["item_description"],
                    "item_category_id": item["item_category_id"],
                    "item_category_description": item["item_category_description"],
                    "rating_value": str(random.randint(0, 1)),
                }
            )
        for item in items_sixth_group:
            ratings.append(
                {
                    "user_id": str(user["user_id"]),
                    "user_description": user["user_description"],
                    "item_id": str(item["item_id"]),
                    "item_description": item["item_description"],
                    "item_category_id": item["item_category_id"],
                    "item_category_description": item["item_category_description"],
                    "rating_value": str(random.randint(0, 1)),
                }
            )

    # Generating rating based on simulated user preferences
    # for the second group
    for user in users_second_group:
        for item in items_first_group:
            ratings.append(
                {
                    "user_id": str(user["user_id"]),
                    "user_description": user["user_description"],
                    "item_id": str(item["item_id"]),
                    "item_description": item["item_description"],
                    "item_category_id": item["item_category_id"],
                    "item_category_description": item["item_category_description"],
                    "rating_value": str(random.randint(0, 1)),
                }
            )
        for item in items_second_group:
            ratings.append(
                {
                    "user_id": str(user["user_id"]),
                    "user_description": user["user_description"],
                    "item_id": str(item["item_id"]),
                    "item_description": item["item_description"],
                    "item_category_id": item["item_category_id"],
                    "item_category_description": item["item_category_description"],
                    "rating_value": str(random.randint(4, 5)),
                }
            )
        for item in items_third_group:
            ratings.append(
                {
                    "user_id": str(user["user_id"]),
                    "user_description": user["user_description"],
                    "item_id": str(item["item_id"]),
                    "item_description": item["item_description"],
                    "item_category_id": item["item_category_id"],
                    "item_category_description": item["item_category_description"],
                    "rating_value": str(random.randint(0, 2)),
                }
            )
        for item in items_fourth_group:
            ratings.append(
                {
                    "user_id": str(user["user_id"]),
                    "user_description": user["user_description"],
                    "item_id": str(item["item_id"]),
                    "item_description": item["item_description"],
                    "item_category_id": item["item_category_id"],
                    "item_category_description": item["item_category_description"],
                    "rating_value": str(random.randint(0, 1)),
                }
            )
        for item in items_fifth_group:
            ratings.append(
                {
                    "user_id": str(user["user_id"]),
                    "user_description": user["user_description"],
                    "item_id": str(item["item_id"]),
                    "item_description": item["item_description"],
                    "item_category_id": item["item_category_id"],
                    "item_category_description": item["item_category_description"],
                    "rating_value": str(random.randint(4, 5)),
                }
            )
        for item in items_sixth_group:
            ratings.append(
                {
                    "user_id": str(user["user_id"]),
                    "user_description": user["user_description"],
                    "item_id": str(item["item_id"]),
                    "item_description": item["item_description"],
                    "item_category_id": item["item_category_id"],
                    "item_category_description": item["item_category_description"],
                    "rating_value": str(random.randint(0, 1)),
                }
            )

    # Generating rating based on simulated user preferences
    # for the third group
    for user in users_third_group:
        for item in items_first_group:
            ratings.append(
                {
                    "user_id": str(user["user_id"]),
                    "user_description": user["user_description"],
                    "item_id": str(item["item_id"]),
                    "item_description": item["item_description"],
                    "item_category_id": item["item_category_id"],
                    "item_category_description": item["item_category_description"],
                    "rating_value": str(random.randint(0, 2)),
                }
            )
        for item in items_second_group:
            ratings.append(
                {
                    "user_id": str(user["user_id"]),
                    "user_description": user["user_description"],
                    "item_id": str(item["item_id"]),
                    "item_description": item["item_description"],
                    "item_category_id": item["item_category_id"],
                    "item_category_description": item["item_category_description"],
                    "rating_value": str(random.randint(0, 1)),
                }
            )
        for item in items_third_group:
            ratings.append(
                {
                    "user_id": str(user["user_id"]),
                    "user_description": user["user_description"],
                    "item_id": str(item["item_id"]),
                    "item_description": item["item_description"],
                    "item_category_id": item["item_category_id"],
                    "item_category_description": item["item_category_description"],
                    "rating_value": str(random.randint(4, 5)),
                }
            )
        for item in items_fourth_group:
            ratings.append(
                {
                    "user_id": str(user["user_id"]),
                    "user_description": user["user_description"],
                    "item_id": str(item["item_id"]),
                    "item_description": item["item_description"],
                    "item_category_id": item["item_category_id"],
                    "item_category_description": item["item_category_description"],
                    "rating_value": str(random.randint(4, 5)),
                }
            )
        for item in items_fifth_group:
            ratings.append(
                {
                    "user_id": str(user["user_id"]),
                    "user_description": user["user_description"],
                    "item_id": str(item["item_id"]),
                    "item_description": item["item_description"],
                    "item_category_id": item["item_category_id"],
                    "item_category_description": item["item_category_description"],
                    "rating_value": str(random.randint(0, 1)),
                }
            )
        for item in items_sixth_group:
            ratings.append(
                {
                    "user_id": str(user["user_id"]),
                    "user_description": user["user_description"],
                    "item_id": str(item["item_id"]),
                    "item_description": item["item_description"],
                    "item_category_id": item["item_category_id"],
                    "item_category_description": item["item_category_description"],
                    "rating_value": str(random.randint(4, 5)),
                }
            )

    # Here we will store the content of CSV file to save
    data_to_save = ""
    for rating in ratings:
        data_to_save += (
            rating["user_id"]
            + ";"
            + rating["user_description"]
            + ";"
            + rating["item_id"]
            + ";"
            + rating["item_description"]
            + ";"
            + rating["item_category_id"]
            + ";"
            + rating["item_category_description"]
            + ";"
            + rating["rating_value"]
            + "\n"
        )

    with open(RATINGS_PATH, "w") as file:
        file.write(data_to_save)


def train():
    """
    This function is used to create, train, evaluate and save the model
    """
    # We read all the ratings from the data file
    all_ratings = pd.read_csv(
        RATINGS_PATH,
        sep=";",
        names=[
            "user_id",
            "user_description",
            "item_id",
            "item_description",
            "item_category_id",
            "item_category_description",
            "rating_value",
        ],
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
    rating_train = train_ratings["rating_value"]

    # Getting the list of user ids, items ids and ratings
    # used to test the model
    user_id_test = test_ratings["user_id"]
    item_id_test = test_ratings["item_id"]
    rating_test = test_ratings["rating_value"]

    # Here we define the two input layer for the model:
    # starting from a list of user and items
    # we want to predict the rating that an user
    # will give to a given item
    user_input_layer = keras.layers.Input(shape=[1])
    item_input_layer = keras.layers.Input(shape=[1])

    # The embedding size is defined here
    # TODO: try to understand the meaning of this value
    embedding_size = 11

    # After creating two input layer, we use the output of
    # these layers as the input of this two embedding layers
    # TODO: it is not clear what these layers does
    user_embedding_layer = keras.layers.Embedding(
        output_dim=embedding_size, input_dim=max_user_id + 1, input_length=1
    )(user_input_layer)
    item_embedding_layer = keras.layers.Embedding(
        output_dim=embedding_size, input_dim=max_item_id + 1, input_length=1
    )(item_input_layer)

    # Using the embedding layer as input of flatten layers
    user_flatten_layer = keras.layers.Flatten()(user_embedding_layer)
    item_flatten_layer = keras.layers.Flatten()(item_embedding_layer)

    # Here we concatenate the two layers
    # TODO: need to understand why
    input_vectors = keras.layers.concatenate([user_flatten_layer, item_flatten_layer])

    # Here we dropout some values to avoid overfitting the model
    input_vectors = keras.layers.Dropout(0.2)(input_vectors)

    # From the result of previous layer we create the input
    # for the final layer, the output layer
    x = keras.layers.Dense(64, activation="relu")(input_vectors)

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
        batch_size=32,
        epochs=40,
        validation_split=0.2,
        shuffle=True,
        verbose=2,
    )

    # Getting predictions from the model
    train_predictions = model.predict([user_id_train, item_id_train]).squeeze()
    test_predictions = model.predict([user_id_test, item_id_test]).squeeze()

    # Printing the mean absolute error to evaluate the accuracy of the model
    print(mean_absolute_error(train_predictions, rating_train))
    print(mean_absolute_error(test_predictions, rating_test))

    # Saving the model
    model.save(MODEL_PATH)
    del model


def test():
    """
    this function is used to test the saved model
    """
    # Loading the saved model
    model = keras.models.load_model(MODEL_PATH)

    # We read all the ratings from the data file
    ratings = pd.read_csv(
        RATINGS_PATH,
        sep=";",
        names=[
            "user_id",
            "user_description",
            "item_id",
            "item_description",
            "item_category_id",
            "item_category_description",
            "rating_value",
        ],
    )

    # Getting the list of user ids, items ids and ratings
    user_ids = ratings["user_id"]
    item_ids = ratings["item_id"]
    ratings = ratings["rating_value"]

    # Predicting the ratings
    predictions = model.predict([user_ids, item_ids]).squeeze()

    # Adding the predictions to the original dataset
    # to compare the real ratings with the preticted ones
    compare = pd.DataFrame(
        {
            "user": user_ids,
            "item": item_ids,
            "rating": ratings,
            "predicted": predictions,
        }
    )

    # Saving the result as CSV file
    compare.to_csv(TEST_PATH)


def predict():
    """
    this function is used to predict items user may like
    """
    # Loading the saved model
    model = keras.models.load_model(MODEL_PATH)

    # We read all the ratings from the data file
    ratings = pd.read_csv(
        RATINGS_PATH,
        sep=";",
        names=[
            "user_id",
            "user_description",
            "item_id",
            "item_description",
            "item_category_id",
            "item_category_description",
            "rating_value",
        ],
    )

    # Getting the list of user ids, items ids and ratings
    item_ids = ratings["item_id"].unique()
    item_descriptions = ratings["item_description"].unique()
    user_ids = pd.Series([ratings["user_id"][0]] * len(item_ids))
    user_descriptions = pd.Series([ratings["user_description"][0]] * len(item_ids))

    # Predicting the ratings
    predictions = model.predict([user_ids, item_ids]).squeeze()

    # Adding the predictions to the original dataset
    # to compare the real ratings with the preticted ones
    compare = pd.DataFrame(
        {
            "user": user_ids,
            "user description": user_descriptions,
            "item": item_ids,
            "item description": item_descriptions,
            "prediction": predictions,
        }
    )

    # Saving the result as CSV file
    compare.to_csv(PREDICTIONS_PATH)


# Based on parameter given, we execute a different function
if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "train":
        train()
    elif arg == "test":
        test()
    elif arg == "generate":
        generate()
    elif arg == "predict":
        predict()
