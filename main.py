import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from PIL import Image


def convert_image(path):
    img = Image.open(path)
    img.thumbnail((100, 100), Image.ANTIALIAS)
    return np.asarray(img)


def load_image_dataset(path_dir):
    images = []
    labels = []
    for file in glob.glob(path_dir + "/*.png"):
        img = convert_image(file)
        images.append(img)

        if "fiat" in file:
            labels.append(0)
        elif "ford" in file:
            labels.append(1)
    return (np.asarray(images), np.asarray(labels))


def display_images(images, labels, title="Default"):
    class_names = ["fiat", "ford"]
    plt.title(title)
    plt.figure(figsize=(10, 10))
    grid_size = min(25, len(images))
    for i in range(grid_size):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])


def main():
    (train_images, train_labels) = load_image_dataset(
        os.path.dirname(os.path.realpath(__file__)) + "/data/train"
    )
    (test_images, test_labels) = load_image_dataset(
        os.path.dirname(os.path.realpath(__file__)) + "/data/test"
    )

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                64, kernel_size=8, activation=tf.nn.relu, input_shape=(100, 100, 4)
            ),
            keras.layers.Flatten(),
            keras.layers.Dense(320, activation=tf.nn.relu),
            keras.layers.Dense(180, activation=tf.nn.relu),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(60, activation=tf.nn.relu),
            keras.layers.Dense(2, activation=tf.nn.softmax),
        ]
    )

    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.7, nesterov=True)

    model.compile(
        optimizer=sgd, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(train_images, train_labels, epochs=100)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test accuracy:", test_acc)

    predictions = model.predict(test_images)
    print(predictions)

    display_images(test_images, np.argmax(predictions, axis=1))
    plt.show()


if __name__ == "__main__":
    main()
