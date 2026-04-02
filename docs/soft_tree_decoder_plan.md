# Soft Tree Decoder for MotherNet: Implementation Plan

## Overview

Replace MotherNet's MLP prediction head with a **soft oblique decision tree** decoder. Instead of the transformer predicting weights for a small MLP classifier, it predicts split parameters (A, b) for a soft decision tree (Frosst & Hinton, 2017). Leaf labels are computed deterministically from training data via soft routing — they are not predicted by the network.

**Key insight:** We can decouple the transformer backbone from the decoder by pre-computing summary embeddings from a frozen pretrained MotherNet, then training only the new tree decoder head. This makes the initial validation cheap (~1 GB VRAM) and fast to iterate on.

## Background

### MotherNet Architecture (Current)
1. **Transformer backbone** (emsize=512, 12 layers, ~25M params): Processes (x_train, y_train) as a sequence of tokens.
2. **Summary layer** (attention-based, ~4M params): Aggregates ~1152 per-token embeddings into a single 1024-dim per-dataset vector.
3. **MLP decoder** (~44M params): Maps the 1024-dim summary to weights and biases of a small MLP classifier.

At inference: the predicted MLP is applied to x_test to produce class predictions.

### Soft Decision Tree (Frosst & Hinton, 2017)
A full binary tree of fixed depth where:
- Each inner node i computes: p_i(x) = sigmoid(x * w_i + b_i) — the probability of routing right.
- Leaf probabilities are products of routing decisions along root-to-leaf paths (computed via the `TreeMapping` matrix).
- Leaf labels are static distributions over classes, set as the weighted average of training labels reaching each leaf.

**Parameters:** A weight matrix A of shape (n_nodes, input_dim) and a bias vector b of shape (n_nodes,). For depth 8: n_nodes = 255, n_leaves = 256.

### Why This Approach
- **VRAM constraint (~10 GB):** Full end-to-end training of MotherNet is expensive. Pre-computing embeddings and training only the decoder head sidesteps this.
- **Training data is not a bottleneck:** MotherNet trains on synthetic datasets generated on-the-fly from random MLP/GP priors. We can generate as many as we need.
- **The decoder output is much smaller:** The soft tree needs 255 * 100 + 255 = 25,755 parameters vs. the MLP decoder's much larger weight matrices. This means the decoder head is tiny (~2M params).
- **Soft tree is fully differentiable:** Sigmoid routing allows clean gradient flow from the test-set loss back through the tree structure to the decoder weights.

## Phase 1: Pre-computation (One-Time)

### Goal
Generate and cache synthetic datasets with their frozen MotherNet summary embeddings.

### Pipeline
1. Load a pretrained MotherNet checkpoint (default config: emsize=512, 12 layers).
2. Freeze the entire model.
3. For N datasets (target: 100K-500K):
   a. Generate a synthetic dataset from the existing prior bag (96.1% MLP prior + 3.9% GP prior).
   b. Run (x_train, y_train) through the frozen backbone + summary layer to get `train_embed` (1024-dim vector).
   c. Save to disk: `{train_embed, x_train, y_train, x_test, y_test, n_classes}`.
4. Store as a PyTorch Dataset (directory of .pt files or a single memory-mapped file).

### Configuration
| Parameter | Value | Notes |
|-----------|-------|-------|
| n_samples | 1152 (1024 + 128) | From MotherNet default config |
| eval_position | 95% (~1094 train, ~58 test) | Standard train/test split |
| max_features | 100 | Zero-padded if fewer |
| n_classes | 2-10 (random per dataset) | Matches MotherNet training |
| tree_depth | 8 (fixed) | 255 inner nodes, 256 leaves |

### Storage Estimates
| N datasets | Approx. disk space |
|------------|-------------------|
| 100K | ~50 GB |
| 250K | ~125 GB |
| 500K | ~250 GB |

Dominant cost is x_train (~1094 * 100 floats = ~438 KB per dataset). The 1024-dim embedding itself is ~4 KB.

### Implementation Notes
- Reuse the existing `PriorDataLoader` and `get_model()`/`load_model()` infrastructure for data generation and model loading.
- Run pre-computation in batches on GPU (backbone inference is fast, no backprop needed).
- Consider storing in half-precision (float16) to halve disk usage.

## Phase 2: Tree Decoder Training

### Trainable Model
A small MLP that maps the frozen summary embedding to tree parameters:

```
Linear(1024, 2048) -> GELU -> Linear(2048, 255 * 100 + 255)
```

Output is reshaped into:
- **A**: shape (255, 100) — oblique split weights for each inner node
- **b**: shape (255,) — split thresholds

Total trainable parameters: ~2M.

### Forward Pass (Per Dataset in Batch)
1. Load `(train_embed, x_train, y_train, x_test, y_test)` from pre-computed cache.
2. `train_embed (1024)` -> MLP head -> raw output (25,755).
3. Reshape into A (255, 100) and b (255,).
4. Set these as the `linear.weight` and `linear.bias` of a `NeuralTreeModule`.
5. **Route training data:** `leaf_probs_train = tree.forward(x_train)` -> shape (n_train, 256).
6. **Set leaf labels (no grad):** `tree.set_leaves(leaf_probs_train, y_train_onehot)` — weighted average of training labels per leaf.
7. **Route test data:** `output, leaf_probs_test = tree.forward(x_test, pred=True)` -> shape (n_test, n_classes).
8. **Loss:** Cross-entropy on output vs y_test.
9. **Backprop:** Gradients flow through: loss -> x_test routing -> (A, b) -> MLP head. Leaf labels are detached (no gradient).

### Training Configuration
| Parameter | Suggested Value | Notes |
|-----------|----------------|-------|
| Optimizer | AdamW | |
| Learning rate | 1e-4 to 3e-4 | Sweep this |
| Batch size | 8-32 | Cheap, can go higher |
| Epochs | Sweep | Monitor validation loss |
| VRAM | < 1 GB | Only the decoder + tree forward pass |
| Mixed precision | Not needed | Model is tiny |

### Batching Considerations
The `TreeMapping` module contains fixed binary routing matrices that depend only on tree depth. Since depth is fixed at 8, a single `TreeMapping` instance can be shared across all datasets in a batch. The per-dataset variation is only in (A, b) and the data itself.

For batched forward pass, the shapes become:
- A: (batch, 255, 100), b: (batch, 255)
- x_train: (batch, n_train, 100), x_test: (batch, n_test, 100)
- This requires adapting `NeuralTreeModule.forward()` to handle a batch dimension, or looping over the batch.

### Success Criteria
- Beat a standalone soft tree trained per-dataset (the soft tree implementation in `soft_tree/`).
- Beat RADDT trained per-dataset.
- Beat tree alternating optimization trained per-dataset.
- Approach MotherNet MLP accuracy (within a few percentage points).
- Maximum 256 classes (limited by number of leaves).

## Phase 3: Low-Rank Factorization

Once full-rank training validates the approach, add low-rank factorization to reduce the decoder output size:

- Instead of predicting A (255, 100) directly, predict W_s (255, r) where r << 100.
- A learned shared matrix W_c (r, 100) reconstructs A = W_s @ W_c.
- This mirrors MotherNet's existing `weight_embedding_rank` approach.
- Reduces decoder output from 25,500 to 255 * r (e.g. r=16 -> 4,080 — a 6x reduction).

## Phase 4: Variable Depth

Support trees of varying depth across datasets:

**Option A: Multiple heads.** One decoder head per depth (e.g. depth 2-8), plus a depth selector. Each head outputs (A, b) for its specific tree size.

**Option B: Fixed max depth, mask unused nodes.** Always predict depth-8 parameters, but mask out nodes below the actual depth. The tree "shuts down" subtrees by learning near-zero routing probabilities.

**Option C: Curriculum.** Start training at small depths (e.g. 3-4), gradually increase to 8 as training progresses.

## Phase 5: Future Directions (From Original Plan)

These are longer-term ideas from `docs/main_plan.md` to revisit after validation:

1. **Hard tree decoding:** Explore converting the soft tree to a hard tree at inference time (e.g. by taking argmax routing or increasing an inverse temperature parameter).
2. **Sparsity constraints:** Limit the number of features used per node to at most k, for interpretability.
3. **Novel priors:** Design synthetic dataset generators that produce data better suited to tree-structured classifiers (e.g. axis-aligned splits, hierarchical cluster structures).
4. **End-to-end fine-tuning:** Once the decoder is validated, unfreeze the backbone (or just the summary layer) and fine-tune end-to-end with a smaller learning rate, if VRAM allows.
5. **Ensemble of trees:** Predict multiple small trees instead of one large tree.

## Implementation Order

| Step | Description | Depends On |
|------|-------------|------------|
| 1 | Pre-computation script: generate & cache datasets with embeddings | Pretrained MotherNet checkpoint |
| 2 | Tree decoder module: MLP head that outputs (A, b) | `soft_tree/` package |
| 3 | Training loop: load cached data, forward pass, loss, backprop | Steps 1 & 2 |
| 4 | Evaluation: compare against per-dataset soft tree, RADDT, MotherNet | Step 3 |
| 5 | Low-rank factorization of A | Step 4 (validated full-rank) |
| 6 | Variable depth support | Step 5 |

## References

- Frosst & Hinton (2017). *Distilling a Neural Network Into a Soft Decision Tree.* arXiv:1711.09784. See `docs/soft_tree.pdf`.
- MotherNet paper. See `docs/mothernet_paper.pdf`.
- Soft tree implementation: `soft_tree/SoftTree.py`, `soft_tree/NeuralTreeMapping.py`.
- MotherNet decoder: `ticl/models/decoders.py` (MLPModelDecoder class).
- Original stream-of-thought plan: `docs/main_plan.md`.
