""" Train my Anime Model """

# Imports
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from model.anime_model import AnimeCharacterCNN
from pathlib import Path

# Constants
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32
num_classes = 10

# Path to data
ROOT = pathlib.Path().absolute()
DATA_PATH = ROOT / "data"


def load_data():
    """Load data from data folder"""
    # Data transformations
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Load data
    train_dataset = ImageFolder(DATA_PATH / "train", transform=transform)
    test_dataset = ImageFolder(DATA_PATH / "test", transform=transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def initalize_model(num_classes):
    """Create Anime Machine Learning Model"""
    model = AnimeCharacterCNN(num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    summary(model, (3, 128, 128))
    return model, device


def train_model(
    model, dataloader, num_epochs, device, save_path="model/anime_model.py"
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        printf(f"Epoch: {epoch} Loss: {loss.item()}")

        torch.save(model.state_dict(), save_path)


def main():
    """Main function"""
    train_loader, test_loader = load_data()
    model, device = initalize_model(num_classes)
    train_model(model, train_loader, NUM_EPOCHS, device)


if __name__ == "__main__":
    main()
