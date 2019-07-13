import sys
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

from skimage import io
from skimage import transform

from datetime import datetime
from tensorflow import keras
from PIL import Image, ImageDraw
from matplotlib import cm


def get_windows_from_image(path):
    image = Image.open(path)
    width, height = image.size

    windows = []

    windows_size = [60]
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
    return np.asarray(image)


def convert_image(path):
    img = Image.open(path).convert("L")
    img.thumbnail((60, 60), Image.ANTIALIAS)
    return np.asarray(img)


def load_image_dataset(path_dir):
    images = []
    labels = []
    for file in glob.glob(path_dir + "/*.jpg"):
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


def train():
    (train_images, train_labels) = load_image_dataset(
        os.path.dirname(os.path.realpath(__file__)) + "/data/train"
    )
    (test_images, test_labels) = load_image_dataset(
        os.path.dirname(os.path.realpath(__file__)) + "/data/test"
    )

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_arrays = train_images.reshape(-1, 60, 60, 1)
    test_arrays = test_images.reshape(-1, 60, 60, 1)

    model = keras.models.Sequential()

    model.add(
        keras.layers.Conv2D(
            32, kernel_size=5, strides=1, activation="relu", input_shape=(60, 60, 1)
        )
    )
    model.add(keras.layers.MaxPooling2D(pool_size=5))
    model.add(keras.layers.Conv2D(64, kernel_size=3, activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=3, strides=3))
    model.add(keras.layers.Conv2D(128, kernel_size=2, activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(2, activation="softmax"))

    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.7, nesterov=True)

    model.compile(
        optimizer=sgd, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(train_arrays, train_labels, epochs=50)

    test_loss, test_acc = model.evaluate(test_arrays, test_labels)
    print("Test accuracy:", test_acc)

    predictions = model.predict(test_arrays)
    print(predictions)

    display_images(test_images, np.argmax(predictions, axis=1))
    plt.show()

    path = os.path.dirname(os.path.realpath(__file__)) + "/data/models/model"
    model.save(path)
    del model


def test():
    model_path = os.path.dirname(os.path.realpath(__file__)) + "/data/models/model"
    model = keras.models.load_model(model_path)

    image_path_input = (
        os.path.dirname(os.path.realpath(__file__)) + "/data/images/image_input.jpg"
    )
    image_path_output = (
        os.path.dirname(os.path.realpath(__file__)) + "/data/images/image_output.jpg"
    )
    windows = get_windows_from_image(image_path_input)

    image = Image.open(image_path_input).convert("L")

    draw = ImageDraw.Draw(image)

    windows_found = []

    for window in windows:

        cropped_image = image.crop(window)
        converted_image = convert_image_in_memory(cropped_image)

        test_array = converted_image
        test_array = test_array.reshape(-1, 60, 60, 1)
        test_array = test_array / 255.0

        prediction = model.predict(test_array)

        if prediction[0][0] > 0.9993:
            windows_found.append((window, "black"))
            print("Found Fiat logo at", window, prediction[0][0])
        if prediction[0][1] > 0.9993:
            windows_found.append((window, "white"))
            print("Found Ford logo at", window, prediction[0][1])

    for window_found in windows_found:
        draw.rectangle(window_found[0], window_found[1])
    image.save(image_path_output)


def generate():

    images_path = os.path.dirname(os.path.realpath(__file__)) + "/data/train"
    files = glob.glob(images_path + "/*.jpg")

    for file in files:

        # image_output_path = (
        #     file.replace(".jpg", "")
        #     + "_transform_right_"
        #     + str(datetime.now().timestamp()).replace(".", "")
        #     + ".jpg"
        # )

        if "rotate" in file:
            continue

        image_output_path = (
            file.replace(".jpg", "")
            + "_transform_rotate_plus_10_"
            + str(datetime.now().timestamp()).replace(".", "")
            + ".jpg"
        )

        image_input = Image.open(file)

        # prospective = transform.ProjectiveTransform(
        #     np.array(
        #         [
        #             [0.62796, -0.00625, 0.375],
        #             [-0.13653, 0.71634, 8.375],
        #             [-0.00447, -0.00021, 1],
        #         ]
        #     )
        # )
        # projected = transform.warp(
        #     image_input, prospective, output_shape=(70, 70), cval=0.5
        # )
        # scaled = transform.resize(projected, (65, 65))
        # cropped = scaled[0:60, 0:60]

        # io.imsave(image_output_path, cropped)

        rotate = transform.rotate(np.asarray(image_input), 10, cval=0.5)

        io.imsave(image_output_path, rotate)


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "train":
        train()
    elif arg == "test":
        test()
    elif arg == "all":
        train()
        test()
    elif arg == "generate":
        generate()
