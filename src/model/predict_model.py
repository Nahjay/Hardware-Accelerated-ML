# Predict a class based on a trained model and a given image.
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(32, 32))

    # convert to array
    img = img_to_array(img)

    # reshape into a single sample with 3 channels
    img = img.reshape(1, 32, 32, 3)

    # prepare pixel data
    img = img.astype("float32")
    img = img / 255.0
    return img


# load an image and predict the class
def predict_class(filename):
    # load the image
    img = load_image(filename)

    # load model
    model = load_model("model.h5")

    # evaluate the model
    predictions = model.predict(img)

    # predict the class
    predicted_classes = predictions.argmax(axis=-1)

    # Create a hash table to map the predicted class to the actual class
    class_names = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

    # Print the predicted class based on the hash table
    print("Predicted class is: ", class_names[predicted_classes[0]])


# entry point, run the example
if __name__ == "__main__":
    # run the example
    predict_class("sample_image.jpg")
