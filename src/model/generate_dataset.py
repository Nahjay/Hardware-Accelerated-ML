import tensorflow as tf
from tensorflow import keras
from keras import layers


class prepare_dataset:
    def init(self):
        pass

    def load_dataset(self):
        # Load the CIFAR-10 dataset
        (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()

        # enocde target values
        train_y = keras.utils.to_categorical(train_y)
        test_y = keras.utils.to_categorical(test_y)

        return train_x, train_y, test_x, test_y

    def preprocess_pixels(self, train, test):
        # convert from integers to floats
        train_norm = train.astype("float32")
        test_norm = test.astype("float32")

        # normalize to range 0-1
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0

        # return normalized images
        return train_norm, test_norm
