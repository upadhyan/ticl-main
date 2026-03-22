# MotherNet-DTSemNet: Research & Implementation Plan

> **Project:** Adapting MotherNet's hypernetwork framework to output oblique decision trees
> using the DTSemNet/RADDT encoding, in place of its current MLP child network.
> **Goal:** A meta-learned hypernetwork that produces a hard, interpretable oblique
> decision tree in a single forward pass via in-context learning.

---

## Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Key Codebases & Papers](#2-key-codebases--papers)
3. [Architecture Reference](#3-architecture-reference)
   - 3.1 [MotherNet Architecture](#31-mothernet-architecture)
   - 3.2 [DTSemNet Encoding](#32-dtsemnet-encoding)
   - 3.3 [Proposed Hybrid: MotherNet-ODT](#33-proposed-hybrid-mothernet-odt)
4. [Parameter & Feasibility Analysis](#4-parameter--feasibility-analysis)
5. [Dead-Leaf Gradient Problem](#5-dead-leaf-gradient-problem)
6. [Proposed Mitigations](#6-proposed-mitigations)
7. [Phased Implementation Plan](#7-phased-implementation-plan)
8. [Open Research Questions](#8-open-research-questions)

---

## 1. Background & Motivation

**MotherNet** (Mueller et al., ICLR 2025) is a hypernetwork transformer trained on
synthetic tabular classification tasks. Given a training set as context, it produces
the weights of a small MLP child network in a single forward pass -- no dataset-specific
gradient descent required. This is in-context learning for model generation.

**DTSemNet** (Panda et al., ECAI 2024) is a semantically equivalent, invertible encoding
of hard oblique decision trees (ODTs) as neural networks. It uses only ReLU activations
and fixed routing weights, enabling vanilla gradient descent without straight-through
estimator (STE) approximations. The encoding is exact: the argmax output of DTSemNet
equals the leaf selected by the corresponding DT on every input.

**The core idea:** Replace MotherNet's MLP child network with a DTSemNet child network.
The transformer backbone remains unchanged. The decoder predicts the oblique split
hyperplanes of the DT instead of MLP weight matrices. The result is a system that
produces a hard, fully interpretable oblique decision tree from a training set in a
single forward pass.

This is directly relevant to the interpretable ML agenda (cf. Literati/SGT), as it
provides a fast, tuning-free alternative induction mechanism for decision-tree-family
models on small tabular datasets.

---

## 2. Key Codebases & Papers

### Codebases

| Repo | URL | Notes |
|---|---|---|
| `microsoft/ticl` | https://github.com/microsoft/ticl | MotherNet, GAMformer, TabFlex. PyTorch. Training via `python fit_model.py mothernet`. Multi-GPU via `torchrun`. |
| `CPS-research-group/dtsemnet` | https://github.com/CPS-research-group/dtsemnet | DTSemNet reference implementation. PyTorch. Core model in `src/dtsemnet.py`. |

### Papers

| Paper | Venue | Key Reference |
|---|---|---|
| MotherNet: Fast Training and Inference via Hyper-Network Transformers | ICLR 2025 | https://arxiv.org/abs/2312.08598 |
| Vanilla Gradient Descent for Oblique Decision Trees (DTSemNet) | ECAI 2024 | https://arxiv.org/abs/2408.09135 |
| TabPFN | NeurIPS 2022 Workshop | Base architecture for MotherNet transformer backbone |
| GAMformer | ICLR 2025 | Companion model in `ticl`; outputs GAM shape functions via ICL |

### Framework

Both codebases are **PyTorch**. MotherNet's training uses `torchrun` for multi-GPU,
MLFlow for experiment tracking (optional, via `MLFLOW_HOSTNAME` env var).

---

## 3. Architecture Reference

### 3.1 MotherNet Architecture

**Inputs:** Training set $\{(x_i, y_i)\}_{i=1}^N$, each $x_i \in \mathbb{R}^{100}$ (zero-padded),
$y_i \in \{0,\ldots,9\}$ (one-hot encoded).

**Transformer backbone:** 12 layers, embedding size $m = 512$. Identical to TabPFN.
Attention is masked so training tokens attend to all training tokens; test tokens
attend only to training tokens.

**Dataset embedding:** Class-average pooling reduces all token embeddings to a single
vector $E \in \mathbb{R}^{512 \times 10}$ (i.e., one 512-dim embedding per class, concatenated).

**Decoder:** One-hidden-layer MLP: $E \mapsto \mathbb{R}^{4096} \mapsto \mathbb{R}^{|\phi|}$.
Currently $|\phi| = 25{,}738$.

**Child network forward pass (current MLP):** Low-rank weight decomposition with
rank $r = 32$, hidden size $h = 512$:

- $W^p_1, W^p_2 \in \mathbb{R}^{h \times r}$, $W^p_3 \in \mathbb{R}^{N_\text{cls} \times r}$ -- predicted by transformer.
- $W^f_1 \in \mathbb{R}^{r \times d}$, $W^f_2 \in \mathbb{R}^{r \times h}$ -- meta-learned, fixed during ICL.

Forward pass:
```
h1 = relu(W1p @ W1f @ x)
h2 = relu(W2p @ W2f @ h1)
logits = W3p @ h2
```

**Total parameters:** 89M (63M in the decoder).
**Training:** Single A100 80GB, ~4 weeks, cosine annealing LR 3e-5, batch sizes 8/16/32.

**At inference on dataset with $r$ features and $c$ classes:**
- Trim $W^f_1$ to first $r$ rows.
- Trim $W^p_3$ to first $c$ columns.
- Hidden size (512) and rank (32) are fixed.

---

### 3.2 DTSemNet Encoding

For a balanced binary tree of depth $d$:
- $k = 2^d - 1$ internal nodes.
- $2^d$ leaves.

**Trainable parameters only:** For each internal node $i$, an oblique split
$(A_i \in \mathbb{R}^n,\ b_i \in \mathbb{R})$. Total:

$$|\phi| = (2^d - 1)(n + 1)$$

**Network structure (4 layers total, fixed depth regardless of $d$):**

| Layer | Size | Activation | Weights |
|---|---|---|---|
| Input | $n + 1$ | -- | -- |
| Hidden 1 (decision layer) | $k$ | Linear | **Trainable**: $A_i$, $b_i$ per node |
| Hidden 2 (indicator layer) | $2k$ | ReLU | Fixed: $+1$ to $\top_i$, $-1$ to $\bot_i$ |
| Hidden 3 (leaf layer) | $2^d$ | Linear | Fixed: 0/1 routing matrix $R \in \{0,1\}^{2^d \times 2k}$ |
| Output (class layer) | $n_\text{cls}$ | MaxPool | Fixed: 0/1 class assignment mask |

**Routing matrix $R$ construction:** For leaf $T_j$ and internal node $T_i$:
- Weight from $\top_i$ to $L_j$ is 0 if $T_j$ is a left descendant of $T_i$, else 1.
- Weight from $\bot_i$ to $L_j$ is 0 if $T_j$ is a right descendant of $T_i$, else 1.

**Forward pass pseudocode:**
```python
x_aug = torch.cat([x, torch.ones(1)])       # (n+1,)
phi   = phi.reshape(k, n+1)                 # (k, n+1), predicted by transformer

# Layer 1: linear decision values
I    = phi @ x_aug                          # (k,)

# Layer 2: ReLU true/false indicators (fixed ±1 weights)
pos  = F.relu(I)                            # top_i
neg  = F.relu(-I)                           # bot_i
hid  = torch.cat([pos, neg], dim=0)         # (2k,)

# Layer 3: leaf accumulation (fixed routing matrix R)
L    = R @ hid                              # (2^d,), linear

# Layer 4: class aggregation (MaxPool per class, fixed assignment)
logits = segment_amax(L, class_assignment)  # (n_cls,)
```

**Semantic equivalence (Theorem 1, DTSemNet paper):** The leaf $T_\ell$ with all
associated decisions true achieves the strictly maximal value among all leaves.
The argmax of `logits` matches the DT's routing decision on every input.

**Fixed buffers** (`register_buffer` in PyTorch, not meta-trained):
- Routing matrix $R$
- Class assignment mask

---

### 3.3 Proposed Hybrid: MotherNet-ODT

**What changes:**

| Component | MotherNet (current) | MotherNet-ODT (proposed) |
|---|---|---|
| Transformer backbone | Unchanged | Unchanged |
| Dataset embedding | Unchanged | Unchanged |
| Decoder output size | 25,738 | $(2^d - 1)(n+1)$ |
| Low-rank decomposition | Yes (rank 32) | Likely unnecessary (small $|\phi|$) |
| Child forward pass | MLP (3 effective layers) | DTSemNet (4 fixed layers) |
| Fixed child weights | $W^f_1$, $W^f_2$ | Routing matrix $R$, class mask |
| Output | Softmax logits | MaxPool logits |
| Interpretable output | No | Yes: hard oblique DT |

**What is unchanged:** Loss function (cross-entropy), training loop, meta-training
prior (TabPFN synthetic prior), ensembling strategy, preprocessing.

---

## 4. Parameter & Feasibility Analysis

### $|\phi|$ vs. Depth

| Depth $d$ | Internal nodes $k$ | $|\phi|$ ($n=100$) | Leaves | MotherNet decoder capacity |
|---|---|---|---|---|
| 4 | 15 | 1,515 | 16 | Well within (17x headroom) |
| 5 | 31 | 3,131 | 32 | Well within (8x headroom) |
| 6 | 63 | 6,363 | 64 | Comfortable (4x headroom) |
| 7 | 127 | 12,827 | 128 | Comfortable (2x headroom) |
| **8** | **255** | **25,755** | **256** | **~= current decoder output size** |

Depth 8 matches the current decoder almost exactly -- a natural architectural ceiling
with no decoder changes required.

### Expected Points per Leaf

The binding empirical constraint at depth $d$ with $N$ training points:

$$\bar{n}_\text{leaf}(d) = N / 2^d$$

| Depth | $N=3000$ | $N=1000$ | $N=500$ |
|---|---|---|---|
| 4 | 187 | 62 | 31 |
| 5 | 93 | 31 | 15 |
| **6** | **46** | **15** | **7** |
| 7 | 23 | 7 | 3 |
| 8 | 11 | 3 | 1 |

**Recommendation:** Depth 6 is the primary target. It gives ~46 points/leaf at $N=3000$
and remains above the critical threshold. Depth 7 is exploratory. Depth 8 requires
dataset regime expansion or explicit mitigation.

### Gradient Flow Note

DTSemNet's child network is exactly **4 layers deep regardless of tree depth**. Tree
depth increases the width ($k$) of the decision layer, not the network depth. Vanishing
gradients through the child network are not a concern.

The gradient problem is structural: **hard ReLU routing creates zero gradients for
internal nodes whose subtrees receive no training points in a given episode.**
This is distinct from vanishing gradients and requires different mitigations (see Section 6).

---

## 5. Dead-Leaf Gradient Problem

**Definition:** In a given meta-training episode with training set $\{(x_i, y_i)\}$,
a leaf $T_j$ is dead if no $x_i$ is routed to $T_j$ by the current split hyperplanes.
For dead leaf $T_j$, the gradient of the loss w.r.t. every split parameter $(A_i, b_i)$
on the path from root to $T_j$ is exactly zero (not small -- exactly zero, by the
piecewise-linearity of ReLU).

**Severity vs. depth:** At depth 6 with $N=3000$, under random initialization the
expected fraction of empty leaves under a balanced binary tree is non-trivial if the
transformer initially outputs near-zero hyperplanes (degenerate routing). This is worst
at the beginning of meta-training.

**Why it matters for the hypernetwork:** The transformer must learn to predict split
hyperplanes that route the data across all leaves. If empty leaves produce zero gradient,
the transformer receives no signal to correct those hyperplanes. This can create a
self-reinforcing failure: bad hyperplanes -> empty leaves -> no gradient -> bad hyperplanes.

---

## 6. Proposed Mitigations

Listed by priority (recommended implementation order).

### M1: Episode Stratification (Low friction, implement first)

Modify the meta-training data prior so that each synthetic episode guarantees minimum
leaf coverage. Require at least $m_\text{min}$ points (e.g., 10) per leaf in each
episode, satisfied by stratified sampling over feature quantiles for the guaranteed
portion, with the remainder drawn i.i.d. from the TabPFN prior.

At depth 6, $64 \times 10 = 640$ stratified points out of $N = 3000$ total is only
21% of the dataset -- the synthetic distribution is barely perturbed.

**Implementation:** Modify the dataset sampling function in the TabPFN prior.
No changes to the model architecture or training objective.

---

### M2: Soft-to-Hard Temperature Annealing

Replace the ReLU indicator with a smoothed sigmoid during meta-training:

$$\text{soft-}\top_i(I_i, \tau) = \sigma(I_i / \tau), \quad \text{soft-}\bot_i(I_i, \tau) = \sigma(-I_i / \tau)$$

Anneal $\tau$ from a large value (e.g., $\tau_0 = 10$) to a small value (e.g., $\tau_T = 0.1$)
using a cosine or exponential schedule over meta-training steps.

At large $\tau$, routing is near-uniform and every node receives gradient. At small $\tau$,
routing converges to the hard DTSemNet. At inference, use the hard ($\tau = 0$) DTSemNet exactly.

**Risk:** The transformer may over-adapt to soft routing and produce suboptimal
hyperplanes when hardened. Combine with M1 to reduce this risk.

**Implementation:** ~5 lines of code. Replace `F.relu(I)` and `F.relu(-I)` with
`torch.sigmoid(I / tau)` and `torch.sigmoid(-I / tau)` during training, controlled
by a global schedule.

---

### M3: Balanced Routing Auxiliary Loss

Add a regularization term to the meta-training objective penalizing imbalanced routing.
Let $p_j$ = soft fraction of training points routed to leaf $j$ (computed via soft
routing). The augmented loss is:

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{CE} + \lambda \cdot \mathrm{KL}\!\left(\mathbf{p} \,\|\, \mathbf{u}\right)$$

where $\mathbf{u}$ is the uniform distribution over $2^d$ leaves.
Differentiable through M2's soft routing. $\lambda$ is a hyperparameter (start at $0.01$).

**Rationale:** Incentivizes the hypernetwork to predict well-calibrated hyperplanes
that partition data evenly, directly addressing the cause of dead leaves rather than
just patching the gradient.

**Composability:** M2 + M3 are the recommended core combination.

---

### M4: Stochastic Path Dropout

During meta-training, randomly select a fraction $p_\text{drop}$ of internal nodes per
episode and replace their hard routing with uniform routing (set $I_i \leftarrow 0$
before the ReLU layer). This forces gradient to flow through both children of each
dropped node.

At inference: all nodes are hard. No change to inference behavior.

**Implementation:** Before the indicator layer, apply a Bernoulli mask to $I$:
```python
if self.training:
    drop_mask = torch.bernoulli(torch.full((k,), p_drop))
    I = I * (1 - drop_mask)   # zeroed nodes route 50/50 with zero credit
```

Conceptually similar to dropout but operating at the routing level.

---

### M5: Hierarchical Level-Wise Decoding (Novel, Higher Risk)

Replace the flat MLP decoder with a level-wise autoregressive decoder. At each depth
level $\ell$, the decoder predicts the $2^\ell$ split hyperplanes conditioned on:
- The global dataset embedding $E$.
- The empirical routing distribution induced by levels $0, \ldots, \ell-1$.

The routing-conditioned embedding at level $\ell$ is computed by partitioning the
training set according to splits already predicted and computing a per-subtree embedding.

**Properties:**
- Dead-leaf problem disappears by construction: nodes are only decoded if their
  parent routes data to them.
- Gradient always flows through occupied nodes.
- Connects structurally to the AO* level-by-level expansion in Literati/SGT.

**Complexity:** Requires a more substantial architectural change. The decoder becomes
a recurrent or sequential process over tree levels rather than a single MLP forward pass.
This is the most promising long-term direction but should be implemented after M1--M3
are validated.

---

### Recommended Combination by Phase

| Phase | Mitigations Active |
|---|---|
| Phase 2 (baseline) | M1 only |
| Phase 3 (core) | M1 + M2 + M3 |
| Phase 4 (ablation) | M1 + M2 + M3 + M4 |
| Phase 5 (advanced) | M5 (hierarchical decoder) |

---

## 7. Phased Implementation Plan

### Phase 0: Environment Setup

**Goal:** Reproduce MotherNet results and confirm the codebase is runnable.

- [ ] Clone `microsoft/ticl` and `CPS-research-group/dtsemnet`.
- [ ] Set up conda environment from `environment.yml`.
- [ ] Confirm `python fit_model.py mothernet -L 2` trains without error (can run
  for a few hundred steps to validate, not full 4 weeks).
- [ ] Load pretrained MotherNet checkpoint via `MotherNetClassifier` and confirm
  inference on a standard dataset (e.g., `load_breast_cancer`).
- [ ] Read and understand `src/dtsemnet.py` in the DTSemNet repo. Confirm the
  forward pass matches the description in Section 3.2 of this document.

**Deliverables:** Working environment, baseline MotherNet inference confirmed.

---

### Phase 1: DTSemNet Child Network (Standalone)

**Goal:** Implement the DTSemNet child network as a standalone differentiable PyTorch
module, independent of MotherNet.

- [ ] Implement `DTSemNetChild(nn.Module)` with the following interface:
  ```python
  class DTSemNetChild(nn.Module):
      def __init__(self, depth: int, n_features: int, n_classes: int): ...
      def forward(self, x: Tensor, phi: Tensor) -> Tensor:
          # phi: (batch, k*(n+1)) -- predicted split hyperplanes
          # returns: logits (batch, n_classes)
  ```
- [ ] Precompute and register the routing matrix $R \in \{0,1\}^{2^d \times 2k}$
  as a buffer (`self.register_buffer('R', ...)`).
- [ ] Precompute and register the class assignment mask as a buffer.
- [ ] Implement `build_routing_matrix(depth)` utility function.
- [ ] Implement `segment_amax(values, assignment)` using `torch_scatter` or
  `torch.zeros(...).scatter_reduce_`.
- [ ] Unit test: for a hand-crafted phi and input x, verify that the argmax of
  DTSemNetChild's output matches the expected leaf of the corresponding hard DT.
- [ ] Test that gradients flow through phi correctly (call `.backward()` on the
  cross-entropy loss and check `phi.grad is not None` for all entries).

**Deliverables:** `DTSemNetChild` module, unit tests, gradient check passing.

---

### Phase 2: Decoder Swap in MotherNet

**Goal:** Replace MotherNet's MLP child with DTSemNetChild. Train at depth 4 as
a proof of concept.

**Key code locations in `ticl`:**
- Decoder definition: search for the MLP decoder that maps $E \to \phi$ in the
  MotherNet model class.
- Child forward pass: the section that applies $\phi$ to test inputs.
- Decoder output size: the constant 25,738 (or equivalent variable).

**Changes required:**
- [ ] Add `depth` hyperparameter to MotherNet config. Default: 4.
- [ ] Replace decoder output size: change from 25,738 to $(2^d - 1)(n + 1)$.
- [ ] Remove low-rank decomposition (not needed at this $|\phi|$ scale).
- [ ] Instantiate `DTSemNetChild` and use it in the child forward pass.
- [ ] Implement M1 (Episode Stratification): modify the dataset sampling function
  to guarantee $m_\text{min} = 10$ points per leaf.
- [ ] Confirm that a meta-training run at depth 4 produces decreasing loss.
- [ ] Evaluate on CC-18 small benchmark (30 datasets, 50/50 split, 5 repetitions)
  following the MotherNet evaluation protocol.

**Key metric:** ROC AUC on CC-18 small, compared to standard MotherNet and to
DTSemNet trained per-dataset with gradient descent.

**Deliverables:** Working depth-4 MotherNet-ODT, CC-18 benchmark results.

---

### Phase 3: Scaling to Depth 6 with Full Mitigations

**Goal:** Scale to depth 6 and apply M2 + M3 to address dead-leaf gradient issues.

- [ ] Implement M2 (Temperature Annealing):
  - Add `tau_init`, `tau_final`, `tau_schedule` to config.
  - Inject current $\tau$ into `DTSemNetChild.forward()`.
  - At inference time, always use hard routing ($\tau = 0$).
- [ ] Implement M3 (Balanced Routing Loss):
  - Compute soft leaf occupancy during training forward pass.
  - Add $\lambda \cdot \mathrm{KL}(\mathbf{p} \| \mathbf{u})$ to loss.
  - Add $\lambda$ to config, sweep over $\{0.001, 0.01, 0.1\}$.
- [ ] Train at depth 6 with M1 + M2 + M3.
- [ ] Monitor per-epoch: fraction of dead leaves, average points per live leaf,
  KL term value, CE loss value.
- [ ] Evaluate on CC-18 small and compare to depth 4.

**Ablation study (within Phase 3):**

| Run | M1 | M2 | M3 |
|---|---|---|---|
| A | ✓ | -- | -- |
| B | ✓ | ✓ | -- |
| C | ✓ | -- | ✓ |
| **D** | **✓** | **✓** | **✓** |

Compare dead-leaf fraction and final AUC across runs.

**Deliverables:** Depth-6 model, ablation table, monitoring dashboard.

---

### Phase 4: Depth Sweep & Stochastic Path Dropout

**Goal:** Characterize performance vs. depth trade-off and test M4.

- [ ] Train models at depths 4, 5, 6, 7 under the M1+M2+M3 regime.
- [ ] Record: CC-18 ROC AUC, inference time, dead-leaf fraction, $|\phi|$.
- [ ] Implement M4 (Path Dropout): add `p_drop` to config, apply during training.
- [ ] Test M4 at depth 7 where dead leaves are most severe.
- [ ] Produce depth vs. accuracy and depth vs. inference time plots.

**Deliverables:** Depth sweep results, recommendation for optimal depth.

---

### Phase 5: Hierarchical Level-Wise Decoder (Research Extension)

**Goal:** Implement M5 as a structural architectural innovation.

- [ ] Design the level-wise decoder interface:
  ```python
  class HierarchicalTreeDecoder(nn.Module):
      def forward(self, E: Tensor, X_train: Tensor) -> Tensor:
          # E: dataset embedding (512*10,)
          # X_train: training points for routing conditioning
          # returns: phi (k*(n+1),) -- all split hyperplanes
  ```
- [ ] At each level $\ell$, compute a routing-conditioned embedding by partitioning
  $X_\text{train}$ according to splits predicted at levels $0, \ldots, \ell-1$.
- [ ] Predict the $2^\ell$ hyperplanes at level $\ell$ from $E$ and the
  routing-conditioned embedding.
- [ ] Compare against flat decoder (Phase 2/3) on CC-18 small and depth-6 setting.

**Deliverables:** Hierarchical decoder implementation, comparison to flat decoder.

---

### Phase 6: Evaluation & Paper Writing

**Goal:** Produce a complete experimental comparison suitable for submission.

**Baselines to compare against:**
- Standard MotherNet (MLP child)
- TabPFN (teacher)
- GAMformer (from `ticl`)
- DTSemNet trained per-dataset with gradient descent
- EBM (interpret-ML)
- XGBoost (non-interpretable upper bound)

**Evaluation benchmarks:**
- CC-18 small (30 datasets, primary)
- TabZilla subset (for scalability)

**Metrics:**
- Normalized ROC AUC (primary)
- Inference time (fit + predict)
- Tree depth and number of nodes
- Interpretability: fraction of splits using $\leq k$ features (oblique sparsity)

---

## 8. Open Research Questions

1. **Prior alignment:** Does the TabPFN synthetic prior (based on Bayesian NNs and
   structural causal models) generate datasets whose decision boundaries are well-
   approximated by shallow oblique DTs? If not, the prior may need modification
   to include more oblique-DT-compatible generating processes.

2. **Depth selection at inference:** MotherNet uses a fixed MLP topology. For
   ODT, should depth be fixed at meta-training time, or can the hypernetwork learn
   to produce trees of variable effective depth (e.g., by learning near-zero
   hyperplanes for nodes it wants to prune)?

3. **Ensembling:** MotherNet improves with feature and class permutation ensembling
   (8 models). Does this strategy transfer to the ODT setting? Permuting features
   changes the meaning of the oblique splits, which may or may not improve coverage.

4. **Sparse oblique splits:** Adding an $L_1$ penalty on the predicted hyperplane
   weights $A_i$ during meta-training would encourage axis-aligned or near-axis-
   aligned splits, improving interpretability at the potential cost of accuracy.
   Worth studying as a regularization option.

5. **Regression extension:** DTSemNet-regression requires one STE call. Can the
   hypernetwork predict both the split hyperplanes and the leaf regression
   parameters simultaneously, given the STE approximation in the regression case?

---

## Appendix: Key Constants and Defaults

| Constant | Value | Source |
|---|---|---|
| Max input features | 100 | TabPFN/MotherNet fixed |
| Max classes | 10 | TabPFN/MotherNet fixed |
| Transformer embedding size | 512 | MotherNet |
| Transformer layers | 12 | MotherNet |
| Dataset embedding size | 5120 ($512 \times 10$) | MotherNet |
| Decoder hidden size | 4096 | MotherNet |
| Current decoder output size | 25,738 | MotherNet |
| MLP child hidden size | 512 | MotherNet |
| MLP child rank | 32 | MotherNet |
| Training GPU | A100 80GB | MotherNet |
| Meta-training LR | 3e-5 | MotherNet |
| Meta-training batch sizes | 8, 16, 32 (increasing) | MotherNet |
| Ensembling size | 8 models | MotherNet |
| Max training points (transformer) | ~5000 GPU / 100000 CPU | MotherNet |
| DTSemNet network depth | 4 (always) | DTSemNet |
| DTSemNet frameworks | PyTorch | DTSemNet |
| Recommended target depth | 6 | This analysis |
| $|\phi|$ at depth 6, $n=100$ | 6,363 | This analysis |
| Points/leaf at depth 6, $N=3000$ | ~46 | This analysis |