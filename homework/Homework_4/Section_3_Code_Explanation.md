# Section III: Code Explanation (Problem 3.1)

Problem 3.1 has two parts:

- **Part A:** build a custom 3-layer CNN in PyTorch with `nn.Conv2d`, `nn.MaxPool2d` and `nn.Linear`, then train it on CIFAR-10 using `CrossEntropyLoss` and the Adam optimizer. The code is in [`custom_cnn.py`](custom_cnn.py).
- **Part B:** take a pre-trained ResNet-18, freeze its backbone (`requires_grad = False`), and replace the final classification head so it fits the target dataset. The code is in [`transfer_learning_resnet18.py`](transfer_learning_resnet18.py).

Both scripts use CIFAR-10: 60,000 color images of 32×32 pixels in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The split is 50,000 training images and 10,000 test images.

---

## Part A: `custom_cnn.py`, a custom 3-layer CNN

### 1. Architecture (`CustomCNN`)

The starter template in the assignment has **two** conv layers. Problem 3.1 asks for **three**, so I added `conv3`. Because of that, the input size of the first fully connected layer changes from `64*8*8` to `128*4*4`.

Each conv block runs Conv (3×3, padding 1) → ReLU → MaxPool (2×2, stride 2). A 3×3 conv with padding 1 keeps the width and height the same. The pool then halves them.

| Layer | Operation | Output shape (C × H × W) | Learnable parameters |
|-------|-----------|--------------------------|----------------------|
| Input | CIFAR-10 image | 3 × 32 × 32 | 0 |
| conv1 + ReLU + pool | Conv2d(3→32, 3×3, p=1) | 32 × 16 × 16 | 3·32·9 + 32 = **896** |
| conv2 + ReLU + pool | Conv2d(32→64, 3×3, p=1) | 64 × 8 × 8 | 32·64·9 + 64 = **18,496** |
| conv3 + ReLU + pool | Conv2d(64→128, 3×3, p=1) | 128 × 4 × 4 | 64·128·9 + 128 = **73,856** |
| flatten | `x.view(x.size(0), -1)` | 2048 | 0 |
| fc1 + ReLU + Dropout(0.3) | Linear(2048→256) | 256 | 2048·256 + 256 = **524,544** |
| fc2 | Linear(256→10) | 10 (logits) | 256·10 + 10 = **2,570** |
| **Total** | | | **620,362** |

Design notes:

- **Channels double (32 → 64 → 128) while the spatial size halves.** As the network goes deeper it trades spatial resolution for more feature channels. Each new channel can detect a more abstract feature, such as an edge, then a texture, then part of an object.
- **The 3×3 kernels follow Problem 1.1.** Stacking small 3×3 convs gives a large receptive field with fewer parameters than one big kernel would need.
- **Most of the parameters (about 85%) are in `fc1`.** This shows the point from Problem 1.1: conv layers share their weights across the image, while fully connected layers do not.
- **Dropout(0.3)** randomly zeroes 30% of the `fc1` activations during training, which reduces overfitting. It is switched off automatically in `model.eval()`.
- **`forward` returns raw logits with no softmax.** `nn.CrossEntropyLoss` already combines `LogSoftmax` and `NLLLoss`, so adding a softmax in the model would apply it twice.

### 2. Data pipeline (`get_loaders`)

- **Normalization:** each channel is normalized with the CIFAR-10 mean `(0.4914, 0.4822, 0.4465)` and standard deviation `(0.2470, 0.2435, 0.2616)`. Inputs centered near zero make optimization faster and more stable.
- **Training-only augmentation:** `RandomCrop(32, padding=4)` and `RandomHorizontalFlip()` give the network slightly shifted or mirrored copies of each image, which helps it generalize. The test set is never augmented, so the test accuracy is measured on the original images.
- **DataLoaders:** batch size 128. The training data is shuffled every epoch and the test data is not. One epoch is ⌈50,000 / 128⌉ = **391** weight updates.

### 3. Training loop (`train_one_epoch`)

Each mini-batch goes through the standard five steps (they are commented in the code):

1. `optimizer.zero_grad()` clears the gradients from the previous batch. PyTorch adds new gradients to the old ones by default, so they must be cleared first.
2. `outputs = model(images)` runs the forward pass.
3. `loss = criterion(outputs, labels)` computes the cross-entropy loss.
4. `loss.backward()` runs backpropagation and computes ∂loss/∂w for every parameter.
5. `optimizer.step()` updates the weights with Adam.

`model.train()` turns dropout on. The function adds up the loss and the number of correct predictions (`outputs.argmax(1) == labels`) and returns the average loss and accuracy for the epoch.

### 4. Evaluation (`evaluate`)

- `@torch.no_grad()` turns off gradient tracking, which makes evaluation faster and uses less memory.
- `model.eval()` turns dropout off, so every prediction is the same each time it runs.
- It returns the test loss and test accuracy.

### 5. `main()`

- `torch.manual_seed(42)` makes runs repeatable.
- It uses the GPU if one is available and the CPU otherwise.
- It prints the trainable parameter count, which should be 620,362.
- **Loss and optimizer:** `nn.CrossEntropyLoss()` with `torch.optim.Adam(model.parameters(), lr=1e-3)`. Adam keeps a running estimate of the mean and variance of each parameter's gradient, so every weight gets its own adaptive step size.
- It trains for **15 epochs**, printing the train and test loss and accuracy after each one, then saves the weights to `custom_cnn_cifar10.pth`.

---

## Part B: `transfer_learning_resnet18.py`, transfer learning with ResNet-18

### Idea

ResNet-18 was already trained on ImageNet (1.2 million images, 1000 classes), so its conv layers have already learned general visual features such as edges, textures and shapes. Transfer learning **reuses that backbone as a fixed feature extractor** and trains only a new final layer for our 10 classes. This takes much less compute and data than training from scratch.

### 1. Building the model (`build_model`)

```python
model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
```

- **Loading the pre-trained weights:** the assignment uses `resnet18(pretrained=True)`. That argument is deprecated in torchvision 0.13 and later, so the script uses the newer `weights=` argument, which loads the same ImageNet weights. If the torchvision version is too old for `weights=`, a `try/except` falls back to `pretrained=True`.
- **Freezing the backbone:**
  ```python
  for param in model.parameters():
      param.requires_grad = False
  ```
  With this flag set, autograd computes no gradients for these weights and the optimizer never changes them.
- **Replacing the head:** ResNet-18 ends in `fc = Linear(512 → 1000)` for the 1000 ImageNet classes. We replace it with:
  ```python
  model.fc = nn.Linear(model.fc.in_features, num_classes)   # Linear(512 → 10)
  ```
  A newly created layer has `requires_grad=True` by default, so this new head is the **only** part of the model that trains.

| Part | Parameters | Trainable? |
|------|------------|------------|
| Backbone (conv layers + BatchNorm, residual blocks) | 11,176,512 | No (frozen) |
| New head `fc`: Linear(512→10) | 512·10 + 10 = **5,130** | Yes |

Only about 0.05% of the network's weights are updated.

### 2. Data pipeline (`get_loaders`)

- **`Resize(224)`:** CIFAR images are 32×32, but ResNet-18 was trained on 224×224 images. Upscaling them makes the features the backbone learned at that scale more useful.
- **ImageNet normalization:** mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. The frozen backbone expects inputs with the same statistics it was trained on, so these values are used instead of the CIFAR-10 ones from Part A.
- `RandomHorizontalFlip()` is applied to the training data only, and the batch size is 64 because the 224×224 images use more memory.

### 3. Training and evaluation (`run_epoch`)

A single function handles both training and evaluation:

- If an `optimizer` is passed, the model is in **training mode** (`model.train()`) with gradients enabled. Otherwise it is in **evaluation mode** (`model.eval()`) under `torch.set_grad_enabled(False)`.
- The steps are the same as in Part A: forward pass → loss → `zero_grad` → `backward` → `step`.
- **BatchNorm note:** `model.train()` also puts the frozen backbone's BatchNorm layers into training mode. Their γ/β weights stay frozen, but their running mean and variance statistics still update to match CIFAR-10. This is normal and harmless for feature extraction.

### 4. `main()`

- It prints the trainable parameter count (**5,130**) and the frozen parameter count (**11,176,512**). This confirms that the freezing worked.
- `torch.optim.Adam(model.fc.parameters(), lr=1e-3)` passes **only the head's parameters** to the optimizer, which makes it clear that only the head is trained.
- `CrossEntropyLoss` is used again.
- It trains for **5 epochs**, which is enough because only one linear layer is learning, then saves the weights to `resnet18_transfer_cifar10.pth`.

---

## Comparing the two approaches

| | Part A: Custom CNN | Part B: ResNet-18 transfer |
|---|---|---|
| Starting weights | Random | Pre-trained on ImageNet |
| Input size | 32×32 | 224×224 (resized) |
| Trainable params | 620,362 (all) | 5,130 (head only) |
| Total params | 620,362 | 11,181,642 |
| Normalization | CIFAR-10 stats | ImageNet stats |
| Epochs | 15 | 5 |
| Cost per epoch | Low | Higher (forward pass through the whole ResNet at 224×224) |

The custom CNN learns every feature from scratch, so it needs more epochs. The transfer model starts from strong general features and only has to learn a linear classifier on top of them, so it usually reaches good accuracy in far fewer epochs. The trade-off is a heavier forward pass on each step.

## How to run

```bash
pip install torch torchvision
python custom_cnn.py                   # Part A
python transfer_learning_resnet18.py   # Part B
```

Each script downloads CIFAR-10 into `./data/` the first time it runs, prints the loss and accuracy for every epoch, and saves the trained weights as a `.pth` file. A GPU is recommended, especially for Part B.
