# import torch
# from torchvision import transforms
# from PIL import Image
# from anime_model import AnimeCharacterCNN
# from train_model import load_data

# # from data.data_loader import load_data

# # Load the trained model
# train_loader = load_data()
# num_classes = 10  # Change this to the number of classes in your dataset
# print(f"Number of classes: {num_classes}")
# model = AnimeCharacterCNN(num_classes=num_classes)
# model.load_state_dict(torch.load("anime_model.pth"))
# model.eval()

# # Set device to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Define the image transformation
# transform = transforms.Compose(
#     [
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,)),
#     ]
# )


# # Load and preprocess a sample image
# image_path = "/app/data/airplane.jpg"
# image = Image.open(image_path).convert("RGB")
# input_tensor = transform(image).unsqueeze(0).to(device)

# # Make predictions
# with torch.no_grad():
#     output = model(input_tensor)
#     predicted_class = torch.argmax(output).item()


# # Print the predicted class
# print(f"Predicted class: {predicted_class}")

# # Print what the predicted class corresponds to
# class_names = train_loader.dataset.classes
# print(f"Predicted class name: {class_names[predicted_class]}")
import torch
from torchvision import transforms
from PIL import Image
from image_model import ImageCNN
from train_model import load_data


def predict_image(model, image_path, device):
    # Load and preprocess the image
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Adjust normalization as needed
        ]
    )

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Make predictions
    with torch.no_grad():
        output = model(input_tensor)
        predicted_probalities = torch.softmax(output, dim=1).squeeze(0)

    # # Sort the probablities in decenting order and return the top k
    # k = 5
    # top5_prob, top5_catid = torch.topk(predicted_probalities, k=k)
    # top5_prob = top5_prob.tolist()
    # top5_catid = top5_catid.tolist()

    # # Print the top 5 classes predicted by the model
    # print(f"Top-{k} predicted classes and probabilities:")
    # for i in range(k):
    #     print(f"Class: {top5_catid[i]}, probability: {top5_prob[i]}")

    # return top5_catid[0], top5_prob[0]

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(predicted_probalities, descending=True)

    # Select the top three probabilities and indices
    top_3_probs = sorted_probs[:3]
    top_3_indices = sorted_indices[:3]

    # Convert indices to class names
    train_loader = load_data()
    unique_labels = set(train_loader.dataset.classes)
    class_names = list(unique_labels)

    top_3_classes = [class_names[i] for i in top_3_indices]

    return top_3_classes, top_3_probs


def main():
    # Load the trained model
    train_loader = load_data()
    unique_labels = set(train_loader.dataset.classes)
    num_classes = len(unique_labels)
    print(f"Number of classes: {num_classes}")
    model = ImageCNN(num_classes=num_classes)
    model.load_state_dict(torch.load("image_model4.pth"))
    model.eval()

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Example image path
    image_path = "/app/data/horse.jpg"

    # Predict the image class
    predicted_class = predict_image(model, image_path, device)

    # Print the predicted class
    print(f"Predicted class: {predicted_class}")

    # Print Corresponding Class Name
    # class_names = train_loader.dataset.classes
    # print(f"Predicted class name: {class_names[predicted_class]}")


if __name__ == "__main__":
    main()
