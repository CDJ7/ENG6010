"""
ENG 6010 - Homework 4, Problem 3.1 (Part A)
Custom 3-layer CNN trained on CIFAR-10 with CrossEntropyLoss + Adam.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class CustomCNN(nn.Module):
    """Three conv blocks (Conv -> ReLU -> MaxPool) followed by a 2-layer classifier.

    Input: 3 x 32 x 32 (CIFAR-10)
      conv1: 3   -> 32  , 32x32 -> pool -> 16x16
      conv2: 32  -> 64  , 16x16 -> pool -> 8x8
      conv3: 64  -> 128 , 8x8   -> pool -> 4x4
      fc1  : 128*4*4 -> 256
      fc2  : 256 -> num_classes
    """

    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> 32 x 16 x 16
        x = self.pool(F.relu(self.conv2(x)))   # -> 64 x 8 x 8
        x = self.pool(F.relu(self.conv3(x)))   # -> 128 x 4 x 4
        x = x.view(x.size(0), -1)              # flatten -> 2048
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)                        # raw logits (CrossEntropyLoss applies softmax)
        return x


def get_loaders(batch_size=128):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()              # 1. clear old gradients
        outputs = model(images)            # 2. forward pass
        loss = criterion(outputs, labels)  # 3. compute loss
        loss.backward()                    # 4. backpropagation
        optimizer.step()                   # 5. Adam weight update

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        running_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders(batch_size=128)
    model = CustomCNN(num_classes=10).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 15
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch:2d}/{num_epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"test loss {te_loss:.4f} acc {te_acc:.4f}")

    torch.save(model.state_dict(), "custom_cnn_cifar10.pth")
    print("Model saved to custom_cnn_cifar10.pth")


if __name__ == "__main__":
    main()
