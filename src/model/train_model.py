import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from anime_model import AnimeCharacterCNN  # Assuming you have this model definition

# Constants
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 64
ROOT = pathlib.Path().cwd()
DATA_PATH = ROOT.parent / "data"


def load_data():
    """Load data from STL-10 dataset"""
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Adjust normalization as needed
        ]
    )

    # Define path to the dataset
    dataset_path = DATA_PATH
    print(dataset_path)

    # Create STL-10 train dataset
    train_dataset = STL10(
        root=dataset_path, split="train", download=True, transform=transform
    )

    # Create data loader
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    return train_loader


def initialize_model(num_classes):
    """Create Anime Machine Learning Model"""
    model = AnimeCharacterCNN(num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, device


def train_model(model, dataloader, num_epochs, device, save_path="anime_model.pth"):
    print("Training model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )

    print("Starting training loop...")

    best_loss = float("inf")
    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f"Epoch: {epoch}")
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

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
    unique_labels = set(train_loader.dataset.classes)
    num_classes = len(unique_labels)
    model, device = initialize_model(num_classes)
    train_model(model, train_loader, NUM_EPOCHS, device)


if __name__ == "__main__":
    main()
