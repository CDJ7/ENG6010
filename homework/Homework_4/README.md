# Homework 4: Deep Learning, CNNs & Architectures

ENG 6010: AI in Robotics (Instructor: Dr. Hongbo Zhang). Total: 100 points.

## What's in this folder

| File | Description |
|------|-------------|
| [`HW4.pdf`](HW4.pdf) | The assignment handout (Parts I–III) |
| [`custom_cnn.py`](custom_cnn.py) | **Problem 3.1, Part A:** a custom 3-layer CNN (`Conv2d` / `MaxPool2d` / `Linear`) trained on CIFAR-10 with `CrossEntropyLoss` and Adam |
| [`transfer_learning_resnet18.py`](transfer_learning_resnet18.py) | **Problem 3.1, Part B:** transfer learning with a pre-trained ResNet-18. The backbone is frozen (`requires_grad = False`) and a new 10-class head is trained |
| [`Section_3_Code_Explanation.md`](Section_3_Code_Explanation.md) | A walkthrough of the Section III code: architecture, shapes and parameter counts, data pipeline, training loop, and a comparison of the two approaches |

## Assignment overview

- **Part I: Conceptual & Mathematical Fundamentals (30 pts).** Conv output-size and parameter formulas, Sigmoid/Tanh/ReLU/Leaky ReLU, the proof that two stacked 3×3 convs equal one 5×5 conv, and gradient flow through ResNet residual blocks.
- **Part II: Numerical Calculations (40 pts).** A hand-computed 2D convolution and max pooling, plus layer-by-layer output shapes and parameter counts for a 5-layer CNN.
- **Part III: Programming & Implementation (30 pts).** The two PyTorch scripts in this folder.

## Running the code

```bash
pip install torch torchvision
python custom_cnn.py
python transfer_learning_resnet18.py
```

CIFAR-10 downloads automatically into `./data/` the first time a script runs. Trained weights are saved as `custom_cnn_cifar10.pth` and `resnet18_transfer_cifar10.pth`. The dataset and the weights are git-ignored.
