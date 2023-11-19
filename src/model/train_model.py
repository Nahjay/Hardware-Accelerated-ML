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

# Constants
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32
num_classes = 10

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
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Use the filename as the label
        label = filename

        return image, label


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

    # Define your dataset path
    dataset_path = DATA_PATH / "dataset/dataset"

    # Create a custom dataset
    custom_dataset = CustomDataset(root=dataset_path, transform=transform)

    # Create data loader
    train_loader = DataLoader(
        custom_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    return train_loader


# def load_data():
#     """Load data from data folder"""
#     # Data transformations
#     transform = transforms.Compose(
#         [
#             transforms.Resize((64, 64)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,)),
#         ]
#     )

#     # Define your dataset path
#     dataset_path = DATA_PATH / "dataset/dataset"

#     # Create a custom dataset using DatasetFolder
#     class CustomDataset(DatasetFolder):
#         def __init__(self, root, transform=None, target_transform=None):
#             super(CustomDataset, self).__init__(
#                 root,
#                 loader=None,
#                 extensions=".jpg",
#                 transform=transform,
#                 target_transform=target_transform,
#             )
#             self.imgs = self.samples

#         def __len__(self):
#             return len(self.imgs)

#         def __getitem__(self, index):
#             path, target = self.imgs[index]
#             sample = self.loader(path)
#             if self.transform is not None:
#                 sample = self.transform(sample)
#             if self.target_transform is not None:
#                 target = self.target_transform(target)

#             return sample, target

#     # Load data
#     train_dataset = CustomDataset(root=dataset_path, transform=transform)

#     # Create data loader
#     train_loader = DataLoader(
#         train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
#     )

#     return train_loader


def initalize_model(num_classes):
    """Create Anime Machine Learning Model"""
    model = AnimeCharacterCNN(num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    summary(model, (3, 64, 64))  # Adjust input size based on your model
    return model, device


def train_model(
    model, dataloader, num_epochs, device, save_path="model/anime_model.pth"
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

        print(f"Epoch: {epoch} Loss: {loss.item()}")

    torch.save(model.state_dict(), save_path)


def main():
    """Main function"""
    train_loader = load_data()
    model, device = initalize_model(num_classes)
    train_model(model, train_loader, NUM_EPOCHS, device)


if __name__ == "__main__":
    main()
