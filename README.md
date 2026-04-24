# Self-Pruning Neural Network

## Overview
This project implements a neural network that dynamically prunes its own weights during training using learnable gate parameters. Instead of pruning after training, the model learns which connections are unnecessary and suppresses them during the training process.

## Key Idea
Each weight in the network is associated with a learnable gate value between 0 and 1.  
- Gate ≈ 1 → weight is active  
- Gate ≈ 0 → weight is effectively pruned  

The model is trained using a combination of classification loss and sparsity regularization to encourage many gates to become close to zero.

## Methodology

### Prunable Linear Layer
A custom linear layer is implemented where:
- Each weight has a corresponding gate score
- Gate values are obtained using the sigmoid function
- Effective weights are computed as:
  
  weight × sigmoid(gate_score)

### Loss Function
Total loss is defined as:

Total Loss = CrossEntropyLoss + λ × SparsityLoss

- CrossEntropyLoss: for classification
- SparsityLoss: L1 norm of gate values to encourage pruning

### Dataset
The model is trained and evaluated on the CIFAR-10 dataset, which consists of 60,000 images across 10 classes.

## Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|-------------|
| 0.1    | 55.34       | 1.49        |
| 1.0    | 56.22       | 22.96       |
| 10.0   | 55.76       | 85.96       |

## Observations
- Increasing λ increases sparsity significantly
- At λ = 10.0, the network achieves high sparsity (~86%)
- Accuracy remains relatively stable despite heavy pruning
- The model successfully removes redundant connections

## Gate Distribution
The distribution of gate values shows:
- A large concentration near zero, indicating effective pruning
- A smaller number of important connections with higher gate values

## How to Run

### Install dependencies
```bash
pip install torch torchvision matplotlib
