# DeepLabv3+ Semantic Segmentation in PyTorch

This project implements the DeepLabv3+ architecture for semantic image segmentation using PyTorch. It includes custom loss functions (Dice + BCE, IOU) and leverages a ResNet-50 backbone for feature extraction.

## Project Structure

- [`deeplabv3+.py`](deeplabv3+.py): Main implementation of the DeepLabv3+ model, including the Atrous Spatial Pyramid Pooling (ASSP) module and custom ResNet-50 backbone.
- [`dice_score.py`](dice_score.py): Contains the [`DiceBCELoss`](dice_score.py) class, a combination of Dice loss and Binary Cross Entropy loss for segmentation tasks.
- [`IOU.py`](IOU.py): Contains the [`IOU`](IOU.py) class, a custom Intersection over Union loss function.
- `__init__.py`: Marks the directory as a Python package.

## Model Overview

### DeepLabv3+

DeepLabv3+ is a state-of-the-art architecture for semantic segmentation. It uses:

- **Atrous Convolution**: Expands the receptive field without losing resolution.
- **ASSP (Atrous Spatial Pyramid Pooling)**: Captures multi-scale context by applying atrous convolution with different rates.
- **Encoder-Decoder Structure**: Combines high-level and low-level features for precise segmentation.

### Custom Loss Functions

- **Dice + BCE Loss**: [`DiceBCELoss`](dice_score.py) combines Dice loss (for overlap) and BCE (for pixel-wise classification).
- **IOU Loss**: [`IOU`](IOU.py) computes the Intersection over Union, useful for segmentation evaluation and training.

## Usage

### Model Initialization

```python
from deeplabv3+ import Deeplabv3Plus

model = Deeplabv3Plus(num_classes=NUM_CLASSES)
```

### Loss Functions

```python
from dice_score import DiceBCELoss
from IOU import IOU

dice_bce_loss = DiceBCELoss()
iou_loss = IOU()
```

### Training Loop (Example)

```python
import torch.optim as optim

model = Deeplabv3Plus(num_classes=NUM_CLASSES)
criterion = DiceBCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for images, masks in dataloader:
    outputs = model(images)
    loss = criterion(outputs, masks)
    loss.backward()
    optimizer.step()
```

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- PIL
- tqdm

Install dependencies with:

```sh
pip install torch torchvision numpy pillow tqdm
```

## References

- [DeepLabv3+ Paper](https://arxiv.org/abs/1802.02611)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

Feel free to modify and extend this project for your specific