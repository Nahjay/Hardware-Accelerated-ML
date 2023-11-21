import torch
from torchvision import transforms
from PIL import Image
from anime_model import AnimeCharacterCNN
from train_model import load_data

# from data.data_loader import load_data

# Load the trained model
train_loader = load_data()
unique_labels = set(train_loader.dataset.file_list)
num_classes = len(unique_labels)
print(f"Number of classes: {num_classes}")
model = AnimeCharacterCNN(num_classes=num_classes)
model.load_state_dict(torch.load("anime_model.pth"))
model.eval()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the image transformation
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


# Load and preprocess a sample image
image_path = "/app/data/dataset/dataset/Naruto_Uzumaki.jpg"
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# Make predictions
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = torch.argmax(output).item()


# Print the predicted class
print(f"Predicted class: {predicted_class}")
