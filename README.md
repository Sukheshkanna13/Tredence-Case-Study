# Tredence-Case-Study  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qA2JZKTl2vaVMaL1gn6m5lRFWPB8_5ua?usp=sharing)


# Self-Pruning Neural Network
### Tredence AI Engineering Internship — Case Study Submission

A neural network that **learns to prune itself during training** using 
learnable gate parameters and L1 sparsity regularization — implemented 
from scratch in PyTorch on CIFAR-10.

---

## Core Idea

Instead of post-training pruning, each weight has a paired learnable 
gate score. During training, an L1 penalty drives gates toward zero, 
effectively removing weak connections while the network is still learning.

Total Loss = CrossEntropy(ŷ, y) + λ × Σ sigmoid(gate_scores)

---

## Key Components

| Component | Description |
|-----------|-------------|
| `PrunableLinear` | Custom `nn.Linear` with learnable gate scores |
| `SparsityLoss` | L1 norm of all sigmoid gates across prunable layers |
| `LambdaScheduler` | Curriculum sparsity — λ ramps up over warmup epochs |
| `SelfPruningNet` | CNN backbone + prunable classifier head |

---

## Results

### Results Table

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) | Notes |
|------------|-------------------|--------------------|-------|
| 1e-2 | 89.67 | 98.67 | Strong pruning, slight accuracy drop |
| 5e-2 | 91.55 | 99.02 | Best accuracy with near-complete sparsity |
| 1e-1 | 91.28 | 99.05 | Highest sparsity, accuracy maintained |


> Results table and gate distribution plot will be updated 
> once training completes.

---

## Gate Distribution

> A successful result shows a **bimodal distribution** — large spike 
> at 0 (pruned weights) and a cluster near 1 (active weights).

---

## Setup & Run

```bash
git clone https://github.com/Sukheshkanna13/tredence-ai-casestudy
cd tredence-ai-casestudy
pip install -r requirements.txt
python self_pruning_network.py
```

Or run directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qA2JZKTl2vaVMaL1gn6m5lRFWPB8_5ua?usp=sharing)

---

---

## Approach Highlights

- **Dynamic λ scheduling** — curriculum sparsity for stable training
- **Per-layer sparsity reporting** — shows which layers prune more
- **Hard pruning mask** at eval time for true sparse inference
- **Clean modular code** — each component independently testable
