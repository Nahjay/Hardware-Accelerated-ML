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
from anime_model import AnimeCharacterCNN
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
        predicted_class = torch.argmax(output).item()

    return predicted_class


def main():
    # Load the trained model
    train_loader = load_data()
    unique_labels = set(train_loader.dataset.classes)
    num_classes = len(unique_labels)
    print(f"Number of classes: {num_classes}")
    model = AnimeCharacterCNN(num_classes=num_classes)
    model.load_state_dict(torch.load("anime_model.pth"))
    model.eval()

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Example image path
    image_path = "/app/data/car.jpg"

    # Predict the image class
    predicted_class = predict_image(model, image_path, device)

    # Print the predicted class
    print(f"Predicted class: {predicted_class}")

    # Print Corresponding Class Name
    class_names = train_loader.dataset.classes
    print(f"Predicted class name: {class_names[predicted_class]}")


if __name__ == "__main__":
    main()
