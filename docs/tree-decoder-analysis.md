# Tree Decoder for MotherNet: DTSemNet vs RADDT Analysis

## Goal

Modify MotherNet to predict a decision tree instead of an MLP. Both DTSemNet and RADDT encode oblique decision trees as neural networks using ReLU activations, making them candidates for the "child model" that MotherNet's transformer would output.

## Shared Foundation

Both models represent an oblique binary tree of depth D with:
- $T_B = 2^D - 1$ branch nodes
- 
- $T_L = 2^D$ leaf nodes

### Weight Matrices (Identical in Both)

| Parameter | Shape | Meaning |
|-----------|-------|---------|
| **A** (split weights) | `(T_B, p)` | Hyperplane normal for each branch node |
| **b** (split thresholds) | `(T_B,)` | Threshold for branching test `a_t^T x <= b_t` |
| **theta** (leaf predictions) | `(T_L,)` or `(T_L, p+1)` | Prediction values at each leaf |

In both models, **A and b are gradient-optimized**. Leaf predictions are computed deterministically.

### Forward Pass (Steps 1-3 are identical)

1. **Branch evaluation:** Compute `a_t^T x - b_t` for all branch nodes (a single `nn.Linear`)
2. **ReLU violation encoding:** Split into `ReLU(Ax - b)` and `ReLU(b - Ax)` — two non-negative values per node, one is zero (correct direction), the other is violation magnitude
3. **Path violation accumulation:** For each leaf, sum violations along root-to-leaf path → `U[i,t]` (total violation for sample i reaching leaf t). DTSemNet uses a fixed sparse matrix L; RADDT uses precomputed index tensors. Both compute the same values.
4. **Leaf selection — THE KEY DIFFERENCE:**
   - **DTSemNet (classification):** MaxPool over leaves grouped by class. Exact, no approximation.
   - **RADDT:** Softmin over violations: `S(U[i,t]) = exp(-alpha * U[i,t]) / sum(...)`. Approximate, controlled by scale factor alpha.

## Decision: RADDT

### Why Not DTSemNet

DTSemNet has a **structural leaf-to-class assignment** problem. Each leaf is permanently mapped to a class via a fixed architecture (the L matrix + MaxPool grouping). This assignment is:
- Set by default via round-robin (suboptimal for imbalanced classes)
- Requires manual `custom_leaf` tuning per dataset (see Avila example in `net_train.py:241`)
- Fundamentally incompatible with MotherNet, where the transformer must handle arbitrary datasets without per-dataset configuration

Workarounds (predict the assignment, use soft logits) would modify DTSemNet enough that it effectively becomes RADDT's approach.

### Why RADDT

1. **Data-determined leaf labels.** Any leaf can become any class — the majority class of training samples landing in that leaf. No structural constraints, no per-dataset tuning. This is essential for a hypernetwork that must handle arbitrary datasets.

2. **Smoother gradients for meta-training.** Softmin distributes gradient across all leaves proportionally. In meta-learning, where gradient quality through the child model directly affects transformer learning, this richer signal likely helps.

3. **Manageable trade-offs:**
   - Train-inference mismatch (softmin vs argmin): acceptable with high alpha or by keeping soft inference
   - alpha hyperparameter: a single scalar, can be fixed or annealed over meta-training epochs
   - Precomputed ancestor path tensors: can be computed on-the-fly for a given depth

4. **RADDT's per-dataset training tricks (multi-start, warm-start annealing, CART init) do NOT transfer** to MotherNet and are NOT reasons for choosing RADDT. The choice is purely about the formulation.

## Proposed MotherNet-Tree Forward Pass

```
1. Transformer(training_data) -> phi -> reshape to A, b
2. Route training_data through tree using A, b with HARD decisions (no grad)
   -> majority vote per leaf -> leaf labels c
3. Route test_data through tree using A, b (WITH grad)
   -> ReLU violations -> softmin -> weight by c -> class predictions
4. CrossEntropy(predictions, test_labels) -> backprop through step 3 -> A, b -> transformer
```

### Key Architectural Difference from Standard MotherNet

In current MotherNet, the child MLP is a pure function of test data — training data is consumed entirely by the transformer. In the tree version, **the child model needs training data too** (step 2) to compute leaf labels. The child model becomes a function of both training and test data, mediated by A, b.

### Gradient Flow

- Step 2 is no-grad: leaf labels c are treated as constants
- Step 3 gradients flow: loss -> softmin -> ReLU violations -> A, b -> transformer
- The transformer learns: "if I shift these splits, test samples route differently, landing on leaves with different labels"
- Small discontinuities from step 2 (a training sample flipping leaves) are smoothed out by averaging over many synthetic datasets per batch

## References

- DTSemNet paper: `docs/dtsemnet_paper.pdf` — "Vanilla Gradient Descent for Oblique Decision Trees" (ECAI-2024)
- RADDT paper: `docs/raddt_paper.pdf` — "Differentiable Decision Tree via ReLU+Argmin Reformulation" (NeurIPS 2025)
- MotherNet paper: `docs/mothernet_paper.pdf` — "Fast Training and Inference via Hyper-Network Transformers" (ICLR 2025)
- DTSemNet code: `dtsemnet-main/src/dtsemnet.py`
- RADDT code: `RADDT-main/singleGPUorCPUVersion/src/RADDT.py`
- MotherNet code: `ticl/models/mothernet.py`
