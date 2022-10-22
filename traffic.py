from multiprocessing.dummy import active_children
import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    # alt - 1
    '''
    for root, _, files in os.walk(data_dir):
        # iterate over files
        for file in files:
            if not file.startswith('.'):
                # combine filename & directory
                img = cv2.imread(os.path.join(root, file))
                # resize image
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                # append image & category
                images.append(ing)
                labels.append(int(os.path.basename(root)))

    return (images, labels)
    '''
    '''
    # alt -2
    #iterate over directories
    for category in range(0, NUM_CATEGORIES - 1):
        # combine path & directory
        directories = os.path.join(data_dir, str(category))
        # iterate over files
        for filename in os.listdir(directories):
            # combine filename & directory
            img = cv2.imread(os.path.join(directories, filename))
            # resize image
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2RGB)
            # append image & category
            images.append(img)
            labels.append(category)

    return images, labels
    '''

    # alt-3
    #iterate over directories
    for directory in os.listdir(data_dir):
        # iterate over files
        for file in os.listdir(os.path.join(data_dir, directory)):
            # combine filename & directory
            img = cv2.imread(os.path.join(data_dir, directory, file))
            # resize image
            resized_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            # append image & directory
            images.append(resized_img)
            labels.append(int(directory))
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    
    # alt - 1

    model = tf.keras.models.Sequential([

        # Add convolution layer C1
        tf.keras.layers.Conv2D(
            32, (3, 3), activation = "relu", input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        
        # Add max-pooling layer P1
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Add convolution layer C2
        tf.keras.layers.Conv2D(
            32, (3, 3), activation = "relu", input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Add max-pooling layer P2
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Add convolution layer C3
        tf.keras.layers.Conv2D(
            32, (3, 3), activation = "relu", input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Add max-pooling layer P3
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Add flattening layer F
        tf.keras.layers.Flatten(),

        # Add hidden layer H1 
        tf.keras.layers.Dense(256, activation="relu"),
        
        # Add hidden layer H2
        tf.keras.layers.Dense(256, activation="relu"),

        # Add dropout layer
        tf.keras.layers.Dropout(0.5),

        # Add output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")

    ])

    # alt - 2
    """
    model = tf.keras.models.Sequential()
    model.add(keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation = "relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation = "relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(NUM_CATEGORIES, activation="softmax"))
    """

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    return model


if __name__ == "__main__":
    main()

