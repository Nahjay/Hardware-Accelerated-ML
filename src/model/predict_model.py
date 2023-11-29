# Predict a class based on a trained model and a given image.
import tensorflow as tf
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
    model = load_model("model/model.h5")

    # evaluate the model
    predictions = model.predict(img)

    # predict the class
    predicted_classes = predictions.argmax(axis=-1)

    # Get the second and third highest probability
    second_highest_prob = tf.math.top_k(predictions, k=2).values[0][1]
    third_highest_prob = tf.math.top_k(predictions, k=3).values[0][2]

    # Get the classes from the tensors of the second and third highest probability
    second_highest_prob_class = tf.where(predictions == second_highest_prob)
    third_highest_prob_class = tf.where(predictions == third_highest_prob)

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

    # Map the second and third highest probability to the actual class
    print(second_highest_prob_class)
    print(third_highest_prob_class)

    # Print the predicted class based on the hash table
    print("Predicted class is: ", class_names[predicted_classes[0]])

    # Print the second and third highest probability and their corresponding classes
    print("Second highest probability is: ", second_highest_prob)
    print("Third highest probability is: ", third_highest_prob)

    return class_names[predicted_classes[0]]


# entry point, run the example
if __name__ == "__main__":
    # run the example
    predict_class("sample_image.jpg")
