# Report: Self-Pruning Neural Network
**Tredence AI Engineering Internship — Case Study**
**Author: Sukhesh Kanna S**

---

## 1. Why L1 Penalty on Sigmoid Gates Encourages Sparsity

### The Loss Formulation

Each weight `w_ij` in a `PrunableLinear` layer is paired with a learnable
gate score. The effective weight in the forward pass is:

```
gate          = clamp(gate_score, 0, 1)
pruned_weight = weight × gate
output        = pruned_weight @ x + bias
```

Total training loss:

```
L_total = CrossEntropy(y_hat, y) + λ × SparsityLoss

SparsityLoss = (1/N) × Σ gate_i     [normalized L1 across all gates]
```

---

### Why L1 and Not L2?

| Property | L1 penalty | L2 penalty |
|----------|-----------|-----------|
| Gradient magnitude | Constant **1.0** always | `2g` → 0 as g → 0 |
| Behavior near zero | Keeps pushing to exactly 0 | Slows down, never reaches 0 |
| Result | **True sparsity — exact zeros** | Small values, no pruning |

**L2** — as a gate shrinks toward zero, its gradient `2g` also shrinks.
Pruning pressure vanishes before the gate reaches zero. Weights become
small but structurally the network remains fully connected.

**L1** — gradient is a constant `1.0` regardless of current gate value.
A gate at `0.001` is pushed with identical force as one at `0.9`.
This drives gates to **exactly zero** — true structural pruning.

---

### Why clamp Instead of sigmoid?

During implementation, sigmoid gating revealed a critical limitation:

```
sigmoid(-4) = 0.018   ← saturation floor
sigmoid(-5) = 0.0067
```

As gate scores become very negative, sigmoid's gradient approaches zero
(vanishing gradient). The optimizer cannot push gates below `~0.018`
regardless of lambda — the mechanism silently fails.

`clamp(0, 1)` solves this completely:
- Gradient = **1.0 everywhere** in `[0, 1]`
- No saturation, no floor
- Gates reach **exactly 0.0** under L1 pressure

This was a key engineering discovery during training — switching from
sigmoid to clamp was the fix that made self-pruning work correctly.

---

### Gradient Flow Through Gate Scores

Both `weight` and `gate_scores` are `nn.Parameter` — updated by the
optimizer each step. During backprop, gradients flow through two paths:

```
dL_total / d(gate_score) = dL_CE/d(gate_score)      [task signal]
                         + λ × 1.0                   [sparsity pressure]
```

The second term is constant downward pressure on every gate. The network
must actively resist this pressure via classification loss to keep a gate
open. Only weights that genuinely improve accuracy survive.

---

### Dynamic λ Scheduling (Curriculum Sparsity)

λ ramps from 0 to λ_max over the first `warmup_epochs`:

```
λ(t) = λ_max × min(1.0,  t / warmup_epochs)
```

Starting with full pressure from epoch 1 collapses all gates before
useful features are learned. Curriculum scheduling allows the network
to first establish what matters, then prune what doesn't.

---

## 2. Results

### Results Table

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) | Notes |
|------------|-------------------|--------------------|-------|
| 1e-2 | 89.67 | 98.67 | Strong pruning, slight accuracy drop |
| 5e-2 | 91.55 | 99.02 | Best accuracy with near-complete sparsity |
| 1e-1 | 91.28 | 99.05 | Highest sparsity, accuracy maintained |

### Key Observation

All three lambda values converge to high sparsity (~98-99%) while
maintaining strong classification accuracy (~89-91%). This demonstrates
that the CIFAR-10 classifier head is **highly redundant** — over 98% of
the 2.1M dense parameters can be effectively zeroed without significant
accuracy loss.

Notably, λ=5e-2 achieves the best accuracy (91.55%) at 99.02% sparsity,
suggesting the network benefits from moderate sparsity pressure that
encourages it to consolidate representations into fewer, stronger
connections.

The convergence behavior across all lambdas indicates the clamp-based
gating mechanism is robust — the sparse network that emerges is not
lambda-sensitive but rather reflects the true redundancy structure of
the dense classifier.

---

## 3. Gate Distribution Analysis

![Gate Distribution](results/gate_distribution.png)

### What the Plot Shows

Each histogram shows final gate values across all `PrunableLinear` layers.
The dashed line marks the pruning threshold `(gate < 0.05)`.

**Successful self-pruning produces a bimodal distribution:**

- **Massive spike at 0** — gates driven to zero by L1 pressure.
  These weights contribute less than 5% of their potential value
  and are treated as structurally pruned.

- **Small cluster near 0.8-1.0** — gates actively preserved by the
  classification loss. These weights carry task-critical information
  the network chose to keep.

- **Empty middle region** — no gates linger at intermediate values.
  The competing pressures (task loss vs L1) force each gate toward
  a decisive binary outcome: fully active or fully pruned.

This bimodality is the defining signature of successful self-pruning,
distinguishing it from L2 regularization which produces a smooth
distribution of small-but-nonzero values.

---

### Per-Layer Sparsity (λ = 5e-2)

| Layer | Shape | Sparsity |
|-------|-------|---------|
| FC_0 (PrunableLinear_0) | 4096 → 512 | 99.09% |
| FC_1 (PrunableLinear_1) | 512 → 10 | 72.85% |

FC_0 prunes more aggressively because its 2M+ parameters have high
redundancy — many paths through a 4096-dimensional space carry
overlapping information. FC_1 (512→10) retains more connections
because it maps directly to class logits — each of the 10 output
neurons needs sufficient input signal to discriminate correctly.

---

## 4. Design Decisions Summary

| Decision | What | Why |
|----------|------|-----|
| clamp over sigmoid | Gating function | No saturation floor, constant gradient |
| gate init = 0.5 | Starting point | Equal distance from pruned/active |
| Normalized L1 | Sparsity loss | Scale-independent lambda |
| Warmup scheduling | Lambda ramp | Stable training, better accuracy |
| Threshold = 0.05 | Sparsity metric | Honest — reflects clamp gate floor |
| Prune head only | Architecture | Dense layers = highest redundancy |

---

## 5. Setup & Reproduction

```bash
git clone https://github.com/YOUR_USERNAME/tredence-ai-casestudy
cd tredence-ai-casestudy
pip install -r requirements.txt
python self_pruning_network.py
```

Requires GPU for reasonable training time (~35 mins on T4).
Results saved to `results/gate_distribution.png`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qA2JZKTl2vaVMaL1gn6m5lRFWPB8_5ua?usp=sharing)