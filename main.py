import glob
import os
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from PIL import Image


def get_windows_from_image(path):
    image = Image.open(path)
    width, height = image.size

    windows = []

    windows_size = [100]
    for window_size in windows_size:
        for left in range(0, width + 1):
            if left % 10 != 0:
                continue
            if left + window_size > width:
                continue
            for top in range(0, height + 1):
                if top % 10 != 0:
                    continue
                if left + window_size > width:
                    continue
                windows.append((left, top, left + window_size, top + window_size))

    return windows


def convert_image_in_memory(image):
    image = image
    image.thumbnail((100, 100), Image.ANTIALIAS)
    return np.asarray(image)


def convert_image(path):
    img = Image.open(path).convert("L")
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

    train_arrays = train_images.reshape(-1, 100, 100, 1)
    test_arrays = test_images.reshape(-1, 100, 100, 1)

    model = keras.models.Sequential()

    model.add(
        keras.layers.Conv2D(
            32,
            kernel_size=10,
            strides=(1, 1),
            activation="relu",
            input_shape=(100, 100, 1),
        )
    )
    model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))
    model.add(keras.layers.Conv2D(64, kernel_size=10, activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1000, activation="relu"))
    model.add(keras.layers.Dense(2, activation="softmax"))

    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.7, nesterov=True)

    model.compile(
        optimizer=sgd, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(train_arrays, train_labels, epochs=100)

    test_loss, test_acc = model.evaluate(test_arrays, test_labels)
    print("Test accuracy:", test_acc)

    predictions = model.predict(test_arrays)
    print(predictions)

    display_images(test_images, np.argmax(predictions, axis=1))
    plt.show()

    path = os.path.dirname(os.path.realpath(__file__)) + "/data/models/model"
    model.save(path)
    del model


def foo():
    model_path = os.path.dirname(os.path.realpath(__file__)) + "/data/models/model"
    model = keras.models.load_model(model_path)

    image_path = os.path.dirname(os.path.realpath(__file__)) + "/data/images/image.jpg"
    windows = get_windows_from_image(image_path)

    image = Image.open(image_path).convert("L")

    for window in windows:
        cropped_image = image.crop(window)
        converted_image = convert_image_in_memory(cropped_image)

        test_array = converted_image
        test_array = test_array.reshape(-1, 100, 100, 1)
        test_array = test_array / 255.0

        prediction = model.predict(test_array)

        if prediction[0][0] > 0.97:
            print("Found Fiat log at ", window, prediction[0][0])
        if prediction[0][1] > 0.97:
            print("Found Ford logo at ", window, prediction[0][1])


if __name__ == "__main__":
    main()
