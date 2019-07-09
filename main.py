import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re

import pillow
from pillow import Image


def image_to_eight_bit_greyscale(path):
    img = Image.open(path).convert("L")
    return np.asarray(img)


def load_image_dataset(path_dir):
    images = []
    labels = []

    os.chdir(path_dir)
    for file in glob.glob("*.jpg"):
            img = image_to_eight_bit_greyscale(file)
            images.append(img)
            if re.match("fiat_*.*", file):
                labels.append(0)
            elif re.match("ford_*.*", file):
                labels.append(1)

    return (np.asarray(images), np.asarray(labels))


def display_images(images, labels, title = "Default"):
    class_names = ["fiat", "ford"]
    plt.title(title)
    plt.figure(figsize=(10,10))
    grid_size = min(25, len(images))
    for i in range(grid_size):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels[i]])


def main():
    (train_images, train_labels) = load_image_dataset("./data/train")
    (test_images, test_labels) = load_image_dataset("./data/test")

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(150, 150)),
            keras.layers.Dense(128, activation=tf.nn.sigmoid),
            keras.layers.Dense(16, activation=tf.nn.sigmoid),
            keras.layers.Dense(2, activation=tf.nn.softmax)
        ]
    )

    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.7, nesterov=True)
    model.compile(optimizer=sgd, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(train_images, train_labels, epochs=100)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test accuracy:", test_acc)

    predictions = model.predict(test_images)
    print(predictions)

    display_images(test_images, np.argmax(predictions, axis = 1))
    plt.show()


if __name__ == "__main__":
    main()