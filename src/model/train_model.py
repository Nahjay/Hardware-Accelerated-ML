# Imports
import torch
import os
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets import DatasetFolder
from torchsummary import summary  # Assuming you have torchsummary installed
from PIL import Image
import pathlib
from pathlib import Path
from anime_model import AnimeCharacterCNN
from sklearn.preprocessing import LabelEncoder

# Constants
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 64
# num_classes = 10

# Path to data
ROOT = pathlib.Path().cwd()
DATA_PATH = ROOT.parent / "data"
dataset_path = DATA_PATH / "dataset/dataset"


class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.file_list = [f for f in os.listdir(root) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        image_path = os.path.join(self.root, filename)

        # Handle potential errors when opening and converting images
        try:
            image = Image.open(image_path).convert("RGB")
        except (IOError, OSError) as e:
            print(f"Error opening image: {filename}")
            return None  # Skip the invalid image

        if self.transform is not None:
            image = self.transform(image)

        # Use the filename as the label
        label = filename

        return image, label


#     return train_loader
def load_data():
    """Load data from data folder"""
    # Data transformations
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomVerticalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomPerspective(p=0.2),
            transforms.RandomResizedCrop(64),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Define your dataset path
    dataset_path = DATA_PATH / "dataset/dataset"

    # Create a custom dataset
    custom_dataset = CustomDataset(root=dataset_path, transform=transform)

    # Create data loader once outside the function
    train_loader = DataLoader(
        custom_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    return train_loader


def initalize_model(num_classes):
    """Create Anime Machine Learning Model"""
    model = AnimeCharacterCNN(num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, device


def train_model(model, dataloader, num_epochs, device, save_path="anime_model.pth"):
    print("Training model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )

    label_encoder = LabelEncoder()
    print(f"{label_encoder}")
    print("Label encoder created...")
    print("Starting training loop...")

    best_loss = float("inf")
    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f"Epoch: {epoch}")
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            # Convert labels to numerical indices
            labels = label_encoder.fit_transform(labels)
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"Epoch: {epoch} Loss: {loss.item()}")

        epoch_loss = running_loss / len(dataloader)
        scheduler.step(epoch_loss)

        print(f"Epoch: {epoch} Loss: {loss.item()}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)


def main():
    """Main function"""
    train_loader = load_data()
    unique_labels = set(train_loader.dataset.file_list)
    # for label in unique_labels:
    # print(label)
    test = list(unique_labels)

    # Find position of label in list
    for i in range(len(test)):
        if test[i] == "Naruto_Uzumaki.jpg":
            print(i)

    num_classes = len(unique_labels)
    model, device = initalize_model(num_classes)
    train_model(model, train_loader, NUM_EPOCHS, device)


if __name__ == "__main__":
    main()
