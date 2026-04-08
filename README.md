# Rethinkable MLP for MNIST

A simple PyTorch implementation of a Multi-Layer Perceptron (MLP) with a **"rethink" mechanism** for MNIST digit classification.

The model introduces auxiliary reconstruction and consistency losses to encourage the hidden representations to retain more information about the input, mimicking a form of "reflective" learning.

## ✨ Model Architecture

The model (`rethinkable_mlp`) consists of two forward paths and two "rethink" (adjustment) paths:

- **Forward Path**: `Input(784) → Hidden(512) → Output(10)`
- **Rethink Path**: 
  - `Hidden(512) → Input(784)` (reconstructs the original input)
  - `Output(10) → Hidden(512)` (maps classification logits back to the hidden space)

The total loss is a weighted combination of:
- **CrossEntropyLoss** for classification
- **MSELoss** for input reconstruction (`hidden_rethink` vs original image)
- **MSELoss** for hidden consistency (`ground_rethink` vs `hidden_output`)
