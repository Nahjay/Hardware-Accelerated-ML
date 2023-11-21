""" Create Anime Machine Learning Model """

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# Create Model
class AnimeCharacterCNN(nn.Module):
    def __init__(self, num_classes):
        super(AnimeCharacterCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)  # Adjust dropout rate as needed
        self.fc2 = nn.Linear(128, num_classes)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # self.dropout = nn.Dropout(0.5)  # Adjust dropout rate as needed

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=3,
#             stride=stride,
#             padding=1,
#             bias=False,
#         )
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(
#             out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_channels, out_channels, kernel_size=1, stride=stride, bias=False
#                 ),
#                 nn.BatchNorm2d(out_channels),
#             )

#     def forward(self, x):
#         residual = self.shortcut(x)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += residual
#         out = self.relu(out)
#         return out


# # class AnimeCharacterCNN(nn.Module):
# #     def __init__(self, num_classes):
# #         super(AnimeCharacterCNN, self).__init__()

# #         # Convolutional block 1 with residual connection
# #         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
# #         self.bn1 = nn.BatchNorm2d(64)

# #         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
# #         self.bn2 = nn.BatchNorm2d(128)

# #         self.shortcut = nn.Conv2d(64, 128, kernel_size=1, bias=False)

# #         # Convolutional block 2 with residual connection
# #         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
# #         self.bn3 = nn.BatchNorm2d(256)

# #         self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
# #         self.bn4 = nn.BatchNorm2d(512)

# #         self.shortcut2 = nn.Conv2d(128, 512, kernel_size=1, bias=False)

# #         # Max pooling layers
# #         self.pool = nn.MaxPool2d(2, 2)

# #         # Fully connected layers with dropout
# #         self.fc1 = nn.Linear(512 * 8 * 8, 128)
# #         self.dropout1 = nn.Dropout(0.5)

# #         self.fc2 = nn.Linear(128, num_classes)
# #         self.dropout2 = nn.Dropout(0.5)

# #     def forward(self, x):
# #         # Convolutional block 1 with residual connection
# #         residual = self.shortcut(x)
# #         x = self.pool(F.relu(self.bn1(self.conv1(x))))
# #         x = F.relu(self.bn2(self.conv2(x)) + residual)

# #         # Convolutional block 2 with residual connection
# #         residual = self.shortcut2(x)
# #         x = self.pool(F.relu(self.bn3(self.conv3(x))))
# #         x = F.relu(self.bn4(self.conv4(x)) + residual)

# #         # Flatten the output
# #         x = x.view(-1, 512 * 8 * 8)

# #         # Fully connected layer 1 with dropout
# #         x = F.relu(self.fc1(x))
# #         x = self.dropout1(x)

# #         # Fully connected layer 2 with dropout
# #         x = self.fc2(x)
# #         x = self.dropout2(x)

# #         return x


# class AnimeCharacterCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(AnimeCharacterCNN, self).__init__()

#         # Convolutional block 1 with residual connection
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)

#         # self.res_block1 = ResidualBlock(64, 128, stride=2)

#         # Convolutional block 2 with residual connection
#         # self.res_block2 = ResidualBlock(128, 256, stride=2)

#         # Max pooling layers
#         self.pool = nn.MaxPool2d(2, 2)

#         # Fully connected layers with dropout
#         self.fc1 = nn.Linear(256 * 8 * 8, 128)
#         self.dropout1 = nn.Dropout(0.5)

#         self.fc2 = nn.Linear(128, num_classes)
#         self.dropout2 = nn.Dropout(0.5)

#     def forward(self, x):
#         # Convolutional block 1 with residual connection
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         # x = self.res_block1(x)

#         # Convolutional block 2 with residual connection
#         # x = self.res_block2(x)

#         # Flatten the output
#         x = x.view(-1, 256 * 8 * 8)

#         # Fully connected layer 1 with dropout
#         x = F.relu(self.fc1(x))
#         x = self.dropout1(x)

#         # Fully connected layer 2 with dropout
#         x = self.fc2(x)
#         x = self.dropout2(x)

#         return x
