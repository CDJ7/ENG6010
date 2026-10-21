"""
ENG 6010 - Homework 4, Problem 3.1 (Part B)
Transfer learning with a pre-trained ResNet-18:
  - backbone (feature extractor) weights frozen with requires_grad = False
  - final fully connected head replaced to fit the target dataset (CIFAR-10, 10 classes)
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def build_model(num_classes=10):
    # The assignment specifies resnet18(pretrained=True). That argument is deprecated
    # in torchvision >= 0.13; the `weights=` form below loads the same ImageNet weights.
    try:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
    except AttributeError:  # older torchvision
        model = torchvision.models.resnet18(pretrained=True)

    # 1. Freeze every backbone parameter -> used as a fixed feature extractor.
    for param in model.parameters():
        param.requires_grad = False

    # 2. Replace the classification head. ResNet-18's head is fc: Linear(512 -> 1000).
    #    A newly created layer has requires_grad=True by default, so only it trains.
    in_features = model.fc.in_features  # 512
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_loaders(batch_size=64):
    # ImageNet statistics, since the backbone was pre-trained on ImageNet.
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.Resize(224),               # 32x32 CIFAR -> 224x224 expected by ResNet
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2))


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(num_classes=10).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable parameters (new head): {trainable:,}")   # 512*10 + 10 = 5,130
    print(f"Frozen parameters (backbone):    {frozen:,}")

    train_loader, test_loader = get_loaders(batch_size=64)
    criterion = nn.CrossEntropyLoss()
    # Only the head's parameters are passed to the optimizer.
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        te_loss, te_acc = run_epoch(model, test_loader, criterion, device)
        print(f"Epoch {epoch}/{num_epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"test loss {te_loss:.4f} acc {te_acc:.4f}")

    torch.save(model.state_dict(), "resnet18_transfer_cifar10.pth")
    print("Model saved to resnet18_transfer_cifar10.pth")


if __name__ == "__main__":
    main()
