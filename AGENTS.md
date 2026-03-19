# AGENTS.md - Fruit AI Project Guidelines

## Project Overview

This is a Python machine learning project for fruit image classification using PyTorch and OpenCV. The project trains a CNN model to classify fruits (apple, banana, mango) and provides test scripts for image and webcam-based inference.

## Directory Structure

```
fruit-ai/
├── training/
│   └── model_training.py      # CNN model training script
├── test/
│   ├── model_evaluation.py     # Model evaluation with metrics
│   ├── model_image_test.py     # Test model on images in folder
│   ├── model_webcam_test.py    # Real-time webcam classification
│   └── webcam_test.py          # Basic webcam preview
├── dataset/
│   ├── fruits/                 # Training images
│   │   ├── apple/
│   │   ├── banana/
│   │   └── mango/
│   └── test/                   # Test images
├── model/
│   └── fruit_model.pth        # Trained PyTorch model
└── confusion_matrix.png        # Evaluation visualization
```

## Dependencies

- Python 3.x
- PyTorch
- OpenCV (cv2)
- NumPy
- scikit-learn
- matplotlib (for evaluation)
- seaborn (for evaluation)

Install dependencies:
```bash
pip install torch opencv-python numpy scikit-learn matplotlib seaborn
```

## Running Scripts

### Training
```bash
python training/model_training.py
```
- Trains a CNN with 10 epochs
- Saves model to `MODEL_PATH` (configurable in script)
- Dataset loaded from `DATASET_PATH` (configurable in script)

### Evaluation
```bash
python test/model_evaluation.py
```
- Computes accuracy, loss, precision, recall, F1-score per class
- Generates confusion matrix visualization

### Testing Images
```bash
python test/model_image_test.py
```
- Navigate images with `a`/`d` keys
- Press `ESC` to exit

### Webcam Testing
```bash
python test/model_webcam_test.py
```
- Press `q` to quit

### Single Script Execution
```bash
python <path/to/script.py>
```

## Code Style Guidelines

### Imports
- Standard library imports first
- Third-party imports second
- Related imports grouped together
- Use absolute imports where possible
- Alphabetical ordering within groups

```python
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
```

### Formatting
- 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Use blank lines to separate logical sections
- No trailing whitespace

### Naming Conventions
- **Variables/Functions**: `snake_case` (e.g., `image_path`, `load_model`)
- **Classes**: `PascalCase` (e.g., `FruitClassifier`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `MODEL_PATH`)
- **Private members**: Leading underscore (e.g., `_internal_state`)

### Type Annotations
- Add type hints for function parameters and return values where beneficial
- Use `typing` module for complex types (List, Dict, Optional)

```python
def predict_image(path: str) -> tuple[str, float]:
    ...
```

### Error Handling
- Check for `None` values explicitly (e.g., `if img is not None`)
- Use `exit()` for fatal errors in test scripts (acceptable for standalone scripts)
- Add descriptive error messages with context

```python
if img is None:
    print("Error: Could not load image")
    return None
```

### Path Handling
- Use `os.path.join()` for cross-platform path construction
- Use raw strings (`r"path"`) for Windows paths with backslashes
- Consider using `pathlib.Path` for more complex path operations

### Documentation
- Use docstrings for functions and classes
- Keep docstrings concise but informative
- Follow Google-style docstring format:

```python
def train_model(data_path: str, epochs: int = 10) -> None:
    """Train the fruit classification model.

    Args:
        data_path: Path to the dataset directory
        epochs: Number of training epochs (default: 10)

    Returns:
        None
    """
```

### Model Configuration
- Store model paths and dataset paths as module-level constants
- Make paths configurable via constants at the top of files
- Use consistent image dimensions (100x100 in this project)

### Testing Conventions
- Test scripts are interactive utilities, not unit tests
- Use `cv2.waitKey()` for keyboard input
- Press `ESC` or `q` to exit loops
- Display predictions with confidence scores

## Hardcoded Paths

Each script contains configurable path constants at the top:
- `MODEL_PATH`: Path to the trained `.pth` model file
- `DATASET_DIR`: Path to training dataset
- `TEST_DIR`: Path to test images (model_image_test.py)

Update these constants before running on different systems.

## Common Issues

- **Camera not opening**: Webcam index may need adjustment (try 0, 1, 2)
- **No images found**: Check file extensions (.jpg, .png, .jpeg, .jfif)
- **Model not found**: Ensure model is trained and path is correct
- **Import errors**: Verify all dependencies are installed

## Adding New Fruit Classes

1. Create a new folder in `dataset/fruits/<fruit_name>/`
2. Add training images
3. Update `class_names` list in training script
4. Update model output layer in all scripts
5. Retrain the model

## VS Code Tasks

The project includes a VS Code task to activate the Python virtual environment:
- Run via `Ctrl+Shift+B` (triggers default build task)
