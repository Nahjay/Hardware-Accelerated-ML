import torch
from torchvision.transforms import ToTensor
from PIL import Image
from model.anime_model import AnimeCharacterCNN

# Load the trained model
model = AnimeCharacterCNN(num_classes=your_num_classes)
model.load_state_dict(torch.load("model/anime_model.pth"))
model.eval()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the image transformation
transform = ToTensor()

# Load and preprocess a sample image
image_path = "path/to/your/test/image.jpg"
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# Make predictions
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = torch.argmax(output).item()

# Print the predicted class
print(f"Predicted class: {predicted_class}")
