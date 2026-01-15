# BiBE: Bidirectional Bug Exorcist

## Technical Design Document

---
## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ TRACE INGESTION & PREPROCESSING                                 │
├─────────────────────────────────────────────────────────────────┤
│ Raw perf trace → Parse → Sliding window extraction → Normalize  │
│ • Function IDs, timestamps, call depth, cache events, branches  │
│ • Window size W (e.g., 512-2048 events)                         │
│ • Temporal binning for timestamp features                       │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ EMBEDDING LAYER                                                 │
├─────────────────────────────────────────────────────────────────┤
│ • Learned function ID embeddings (vocab size V, dim d_model)    │
│ • Positional encodings (sinusoidal, NOT learned - alibi better) │
│ • Auxiliary feature projection (depth, cache misses, etc.)      │
│ → Output: [batch, seq_len, d_model]                             │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ ATTENTION LAYERS (Stack of L transformer blocks)                │
├─────────────────────────────────────────────────────────────────┤
│ Each block:                                                     │
│   1. Multi-head self-attention (H heads, d_k = d_model/H)       │
│   2. Residual connection + LayerNorm                            │
│   3. Position-wise FFN (2-layer MLP, GeLU activation)           │
│   4. Residual connection + LayerNorm                            │
│                                                                 │
│ Key design: Causal masking DISABLED for full bidirectional      │
│             attention (bugs can be caused by future events)     │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ ATTRIBUTION HEAD                                                │
├─────────────────────────────────────────────────────────────────┤
│ • Per-position classification: P(anomaly | position)            │
│ • Contrastive loss between anomalous and normal traces          │
│ • Attention rollout for causal attribution to root causes       │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUTS                                                         │
├─────────────────────────────────────────────────────────────────┤
│ 1. Per-event anomaly scores                                     │
│ 2. Attention maps (which past events caused this anomaly?)      │
│ 3. Gradient-based saliency (∂loss/∂input features)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Concrete Model Design

### 2.1 Input Representation

Each trace event `e_i` at position `i` consists of:

- **Function ID** (categorical, vocab size V ≈ 1000-10000)
- **Timestamp** (continuous, microseconds)
- **Call depth** (integer, 0-max_depth)
- **Cache events**: L1/L2/LLC misses (counts)
- **Branch mispredictions** (count)
- **Optional**: CPU utilization, page faults, etc.

**Embedding:**

```
x_i = W_func[func_id] + PE(i) + W_aux * [depth, log(cache_misses), log(branches), ...]
```

Where:

- `W_func ∈ ℝ^{V × d_model}`: Learned embedding matrix
- `PE(i)`: Sinusoidal positional encoding (Vaswani et al. 2017 formula)
- `W_aux ∈ ℝ^{n_aux × d_model}`: Projects auxiliary features

**Critical detail:** Normalize timestamps to relative offsets within the window (t - t_0) and apply log-scaling to handle wide dynamic range.

### 2.2 Multi-Head Self-Attention

**Standard scaled dot-product attention:**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Multi-head formulation:**

```
head_h = Attention(XW_h^Q, XW_h^K, XW_h^V)
MultiHead(X) = Concat(head_1, ..., head_H) W^O
```

**Dimensions:**

- `d_model = 256` (start small, can scale to 512)
- `H = 8` heads
- `d_k = d_v = d_model / H = 32`

**No causal masking** – full bidirectional attention. This is critical because:

1. A segfault at position 500 might be caused by memory corruption at position 200 AND a deallocation at position 700.
2. We're analyzing post-mortem traces, not doing autoregressive generation.

**Numerical stability:**

- Subtract max from logits before softmax
- Use `log-sum-exp` trick for gradient computation
- Clip attention weights to [1e-8, 1.0] before log operations

### 2.3 Layer Normalization

```
LayerNorm(x) = γ ⊙ ((x - μ) / √(σ² + ε)) + β
```

Where:

- `μ, σ²`: mean and variance computed across d_model dimension
- `γ, β ∈ ℝ^{d_model}`: learned scale and shift
- `ε = 1e-5`: stability constant

**Pre-norm vs Post-norm:** Use **Pre-LN** (LayerNorm before attention/FFN) for better gradient flow in deep stacks.

### 2.4 Position-wise Feed-Forward Network

```
FFN(x) = W_2 * GeLU(W_1 * x + b_1) + b_2
```

- `W_1 ∈ ℝ^{d_model × d_ff}`, typically `d_ff = 4 * d_model = 1024`
- `W_2 ∈ ℝ^{d_ff × d_model}`

**GeLU activation:**

```
GeLU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

### 2.5 Output Head

Two parallel heads:

**A. Anomaly Detector (per-position binary classification):**

```
logits_i = W_cls * h_i + b_cls    (h_i ∈ ℝ^{d_model})
P(anomaly_i) = sigmoid(logits_i)
```

**B. Causal Attribution (attention rollout):**

Multiply attention matrices across layers to find transitive dependencies:

```
A_rollout = A^(L) · A^(L-1) · ... · A^(1)
```

For event `i` flagged as anomalous, rank source events `j` by `A_rollout[i, j]`.

---

## 3. Learning Objectives & Loss Functions

### 3.1 Primary Loss: Contrastive Trace Loss

**Idea:** Learn to distinguish anomalous traces from normal traces at the sequence level.

Given a batch of traces:

- Positive samples: Traces containing known bugs (segfault, deadlock, perf regression)
- Negative samples: Clean execution traces

**Implementation:**

```
L_contrastive = -log( exp(sim(h_anom, h_bug) / τ) / 
                      (exp(sim(h_anom, h_bug) / τ) + Σ_n exp(sim(h_anom, h_normal_n) / τ)) )
```

Where:

- `h_anom`: Mean-pooled representation of anomalous trace
- `h_bug`, `h_normal_n`: Anchor embeddings
- `sim(·,·)`: Cosine similarity
- `τ = 0.07`: Temperature parameter

**Why this works:** Forces the model to build representations where anomalous patterns cluster together, distinct from normal execution.

### 3.2 Auxiliary Loss: Per-Event Anomaly Detection

For traces with ground-truth anomaly positions (e.g., "crash occurred at line 1243"):

```
L_anomaly = Σ_i BCE(P(anomaly_i), y_i)
```

Where `y_i ∈ {0, 1}` is ground truth label.

**Weighting:** Use focal loss to handle class imbalance (most events are normal):

```
L_focal = -Σ_i α * (1 - p_i)^γ * log(p_i)    if y_i = 1
          -Σ_i (1 - α) * p_i^γ * log(1 - p_i)  if y_i = 0
```

Set `α = 0.75`, `γ = 2.0`.

### 3.3 Regularization: Attention Sparsity

Encourage the model to focus on a small number of causal events:

```
L_sparse = λ * Σ_{i,j} H(A[i,j])
```

Where `H(p) = -p log p` is entropy and `λ = 0.01`.

**Total loss:**

```
L_total = L_contrastive + β * L_anomaly + L_sparse
```

Start with `β = 1.0`, anneal to 0.1 after convergence.

---

## 4. Implementation Roadmap

### **Phase 0: Math & Numerical Core (2-3 weeks)**

**Goal:** Build a correct, numerically stable linear algebra library in Rust.

**Tasks:**

1. Implement dense matrix operations:
    - Matrix multiplication (naive, then cache-blocked)
    - Transpose, element-wise ops
    - Broadcasting semantics
2. Numerical stability primitives:
    - Log-sum-exp
    - Numerically stable softmax
    - Gradient clipping
3. Autograd system:
    - Computation graph with topological sort
    - Reverse-mode autodiff (store intermediate activations)
    - Gradient accumulation and zeroing
4. **Test harness:**
    - Compare against NumPy/PyTorch on toy problems
    - Gradient checking with finite differences (ε = 1e-5)

**Prerequisites:**

- "Numerical Linear Algebra" by Trefethen & Bau (Chapters 1-3)
- Understanding of floating-point arithmetic (IEEE 754)
- Backpropagation calculus (CS231n notes)

**Key risks:**

- **Numerical instability in softmax:** Always subtract max logit
- **Exploding gradients:** Clip by global norm (threshold = 1.0)
- **Memory leaks in autograd:** Use Rust's ownership correctly, or RAII in C++

---

### **Phase 1: Attention Mechanism (2 weeks)**

**Goal:** Implement and validate multi-head self-attention.

**Tasks:**

1. Implement scaled dot-product attention:
    
 ```rust
   fn attention(Q: &Tensor, K: &Tensor, V: &Tensor) -> (Tensor, Tensor) {
       let scores = (Q @ K.transpose()) / (d_k as f32).sqrt();
       let attn_weights = softmax(scores, dim=-1);
       let output = attn_weights @ V;
       (output, attn_weights)  // Return weights for visualization
   }
```
    
2. Implement multi-head attention with learned projections
3. **Validation:**
    - Overfit a tiny synthetic task (e.g., copy mechanism on sequences)
    - Verify attention weights sum to 1.0 per query
    - Check gradient flow through all projections

**Prerequisites:**

- "Attention is All You Need" (Vaswani et al., 2017) – read 3+ times
- Matrix calculus for backprop through matrix multiplies

**Key risks:**

- **Attention collapse:** All weights converge to uniform. Fix: Ensure proper initialization (Xavier/He), check learning rate.
- **NaN in gradients:** Use gradient clipping, check for divide-by-zero.

---

### **Phase 2: Full Transformer Block (2 weeks)**

**Goal:** Stack attention + FFN + LayerNorm into a working encoder layer.

**Tasks:**

1. Implement LayerNorm with learnable parameters
2. Implement position-wise FFN with GeLU
3. Implement residual connections
4. Stack L=4 layers initially
5. **Validation:**
    - Overfit on a sequence classification task (even/odd parity of function IDs)
    - Verify each layer's output has mean ≈ 0, std ≈ 1 (due to LayerNorm)
    - Profile memory usage (activations grow with depth)

**Prerequisites:**

- "Layer Normalization" (Ba et al., 2016)
- Understanding of residual connections and gradient highways

**Key risks:**

- **Vanishing gradients in deep stacks:** Use Pre-LN, monitor gradient norms per layer.
- **Memory explosion:** Store only necessary activations for backprop; recompute others if needed.

---

### **Phase 3: Data Pipeline & Embedding (1-2 weeks)**

**Goal:** Parse real perf traces and convert to model inputs.

**Tasks:**

1. Write a parser for `perf script` output (or your trace format):
	```
	functionID timestamp depth cache_misses branch_misses ...
	```
    
2. Implement sliding window extraction (overlapping or non-overlapping)
3. Build vocabulary of function IDs with special tokens (`<PAD>`, `<UNK>`)
4. Implement sinusoidal positional encodings
5. Create data loader with batching and shuffling

**Prerequisites:**

- Understanding of sequence padding and masking
- Familiarity with perf or similar profiling tools

**Key risks:**

- **Rare function IDs:** Use subword tokenization (BPE) or hash embeddings if vocab is huge.
- **Timestamp normalization:** Wrong normalization destroys temporal signal. Always use relative offsets.

---

### **Phase 4: Training Loop & Optimization (2 weeks)**

**Goal:** Train the model on real traces with proper optimization.

**Tasks:**

1. Implement Adam optimizer from scratch:
    
   ```
    m_t = β1 * m_{t-1} + (1 - β1) * g_t
    v_t = β2 * v_{t-1} + (1 - β2) * g_t²
    θ_t = θ_{t-1} - α * m_t / (√v_t + ε)
     ```
    (β1=0.9, β2=0.999, α=3e-4, ε=1e-8)
    
1. Implement learning rate warmup + cosine decay:
    
   ```
    lr(t) = lr_max * min(t / warmup_steps, 0.5 * 
	    (1 + cos(π * (t - warmup_steps) / T)))
     ```
    
2. Implement gradient clipping by global norm
3. Add checkpointing (save model every N steps)
4. **Validation:**
    - Monitor train/val loss curves
    - Check that gradients don't explode (log gradient norms)
    - Verify Adam state updates correctly

**Prerequisites:**

- "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
- Understanding of learning rate schedules and their impact

**Key risks:**

- **Overfitting:** Use dropout (0.1-0.2) in FFN and attention
- **Learning rate too high:** Model diverges. Start at 1e-4, tune carefully.
- **Insufficient data:** You'll need 10K+ traces minimum. Augment with synthetic bugs.

---

### **Phase 5: Attribution & Evaluation (2-3 weeks)**

**Goal:** Make attention weights interpretable and validate causal claims.

**Tasks:**

1. Implement attention rollout across layers
2. Implement gradient-based saliency (∂L/∂input features)
3. **Quantitative evaluation:**
    - Precision@K: Of top-K attended events, how many are causally relevant?
    - Mean Reciprocal Rank (MRR) of ground-truth root cause
    - AUC for anomaly detection at event level
4. **Qualitative evaluation:**
    - Visualize attention heatmaps for known bugs
    - Show that model attends to null pointer dereference → segfault
    - Show that model attends to lock acquisition → deadlock
5. **Ablation studies:**
    - Remove positional encodings → does performance drop?
    - Use random attention weights → does it still work? (It shouldn't.)

**Prerequisites:**

- "Attention Interpretability Across NLP: A Survey" (Bastings & Filippova, 2020)
- Understanding of precision/recall for ranking problems

**Key risks:**

- **Spurious correlations:** Model might latch onto irrelevant high-frequency functions. Use contrastive negatives carefully.
- **Attention doesn't mean causation:** Validate with controlled synthetic bugs where ground truth is known.

---

## 5. Key Risks & Failure Modes

### 5.1 Numerical Risks

| Risk                    | Symptom                     | Mitigation                               |
| ----------------------- | --------------------------- | ---------------------------------------- |
| **Softmax overflow**    | NaN in attention weights    | Subtract max logit before exp            |
| **Gradient explosion**  | Loss → ∞ after few steps    | Clip gradients by global norm (1.0)      |
| **Vanishing gradients** | Lower layers don't learn    | Use Pre-LN, monitor per-layer grad norms |
| **FP precision loss**   | Attention weights sum ≠ 1.0 | Use double precision for critical ops    |

### 5.2 Modeling Risks

|Risk|Symptom|Mitigation|
|---|---|---|
|**Attention collapse**|All weights uniform|Check initialization, increase learning rate|
|**Overfitting to trace length**|Fails on different window sizes|Train with variable-length sequences|
|**Positional encoding mismatch**|Model can't generalize to unseen positions|Use relative positional encodings (T5-style) or ALiBi|
|**Spurious correlations**|Attends to wrong events|Curate negative examples carefully, use contrastive loss|

### 5.3 Systems Risks

|Risk|Symptom|Mitigation|
|---|---|---|
|**Memory exhaustion**|OOM for seq_len > 1024|Implement gradient checkpointing (recompute activations)|
|**Slow training**|Days per epoch|Use BLAS (OpenBLAS, Intel MKL) via FFI|
|**Data bottleneck**|GPU-equivalent time spent on I/O|Preprocess traces offline, memory-map datasets|

---

## 6. Prerequisite Knowledge Checklist

### Mathematics

- [ ] **Linear algebra:** Eigenvalues, SVD, matrix calculus
- [ ] **Multivariable calculus:** Chain rule, Jacobians, Hessians
- [ ] **Probability:** KL divergence, cross-entropy, information theory basics
- [ ] **Numerical analysis:** Floating-point arithmetic, condition numbers

**Resources:**

- "Matrix Cookbook" by Petersen & Pedersen
- "Deep Learning" by Goodfellow et al. (Chapters 2-4)

### Machine Learning

- [ ] **Backpropagation:** Reverse-mode autodiff, computation graphs
- [ ] **Optimization:** SGD, momentum, Adam, learning rate schedules
- [ ] **Regularization:** Dropout, weight decay, gradient clipping
- [ ] **Transformers:** Self-attention, positional encodings, LayerNorm

**Resources:**

- CS231n lecture notes (especially "Backprop" and "Neural Networks")
- "Attention is All You Need" (Vaswani et al., 2017)
- Andrej Karpathy's "Neural Networks: Zero to Hero" series

### Systems Programming

- [ ] **Memory management:** Stack vs heap, alignment, cache hierarchies
- [ ] **Profiling tools:** perf, gprof, valgrind
- [ ] **Concurrency:** Thread safety (if you parallelize training)
- [ ] **Build systems:** Cargo/CMake for managing dependencies

**Resources:**

- "Computer Systems: A Programmer's Perspective" (Bryant & O'Hallaron)
- Rust Book (if using Rust) or "Effective Modern C++" (if C++)

### Numerical Computing

- [ ] **Floating-point:** IEEE 754, catastrophic cancellation
- [ ] **Stability:** Condition numbers, backward/forward error analysis
- [ ] **BLAS/LAPACK:** Understanding what these libraries provide

**Resources:**

- "Numerical Linear Algebra" (Trefethen & Bau)
- "What Every Computer Scientist Should Know About Floating-Point Arithmetic" (Goldberg)

---

## 7. Evaluation & Visualization

### 7.1 Quantitative Metrics

**A. Anomaly Detection Performance:**

- **AUC-ROC** at event level (per-position anomaly classification)
- **Precision@K:** Of the top-K events flagged, how many are in the ground-truth bug region?
- **Mean Average Precision (MAP):** Accounts for ranking quality

**B. Causal Attribution Accuracy:**

- **Hit@K:** Is the ground-truth root cause in the top-K attended events?
- **Mean Reciprocal Rank (MRR):** Where does the true root cause rank?

**C. Ablation Baseline:**

- Random attention weights (uniform or random)
- Frequency-based heuristic (flag rare functions)
- LSTM baseline (to show transformers are necessary)

### 7.2 Qualitative Visualization

**A. Attention Heatmaps:**

```
         Event indices (source) →
    ┌─────────────────────────────┐
  E │ [color-coded attention]     │
  v │                             │
  e │                             │
  n │                             │
  t │                             │
    │                             │
  i │                             │
  d │                             │
  x │                             │
    └─────────────────────────────┘
```

Use a color scale where:

- **Red:** High attention (causal relevance)
- **Blue:** Low attention
- Highlight ground-truth root cause with a marker

**B. Trace Replay with Attribution:**

Show a terminal UI or web dashboard that:

1. Replays the execution trace step-by-step
2. Overlays real-time anomaly scores
3. Shows causal arrows from high-attention source events to the anomaly
4. Displays gradient saliency on input features (which cache misses mattered?)

**C. Synthetic Bug Validation:**

Create controlled test cases:

```c
// Bug: Use-after-free
char* ptr = malloc(100);
free(ptr);
// ... 500 lines of normal code ...
ptr[0] = 'X';  // ← Crash here
```

Verify that the model:

1. Flags the crash location
2. Attends strongly to the `free(ptr)` call
3. Shows causal link in attention rollout

### 7.3 Convincing a Skeptical Systems Engineer

**Proof points:**

1. **Reproducibility:** Provide seed-able random initialization, deterministic trace replay
2. **Ablation studies:** Show that removing attention destroys performance
3. **Out-of-distribution generalization:** Train on bugs in program A, test on program B
4. **Human evaluation:** Ask engineers to label top-5 attended events as relevant/irrelevant
5. **Statistical significance:** Run 5+ training runs with different seeds, report mean ± std

**Red flags that would invalidate the work:**

- Model only works on the exact training examples
- Attention weights are near-uniform (not learning)
- Performance doesn't beat a simple frequency baseline
- Can't explain false positives/negatives

---

## 8. Additional Design Considerations

### 8.1 Why No Causal Masking?

Standard transformer decoders use causal masking (can't attend to future tokens). We **explicitly avoid this** because:

1. **Post-mortem analysis:** We have the full trace, not generating it token-by-token
2. **Bidirectional causality:** A crash at position `i` might be explained by:
    - Memory allocation at position `i - 100` (past)
    - Deallocation at position `i + 50` (future, in execution order)
3. **Parallel bugs:** Two race conditions at positions `i` and `j` might both contribute

### 8.2 Why Not Use Pretrained Models?

You're implementing from scratch for good reasons:

- **Interpretability:** You need to trust the math, not treat it as a black box
- **Domain mismatch:** Pretrained language models don't understand execution traces
- **Engineering rigor:** Building from scratch teaches you where failure modes hide

### 8.3 Data Requirements

**Minimum viable dataset:**

- 10,000 normal execution traces
- 1,000 anomalous traces with labeled root causes
- Cover at least 3-5 bug classes (memory errors, deadlocks, performance regressions)

**Data augmentation ideas:**

- Inject synthetic bugs into clean traces
- Time-shift trace windows
- Subsample events (test robustness to trace resolution)

### 8.4 When to Use Rust vs C++

**Use Rust if:**

- You value memory safety and want the compiler to catch bugs
- You're comfortable with the ownership model
- You want to interface with modern tooling (Cargo is excellent)

**Use C++ if:**

- You need mature BLAS libraries (Eigen, Armadillo)
- You're more comfortable with manual memory management
- You want maximum control over every allocation

**My recommendation:** **Rust** for the autograd/training code (safety matters), with FFI bindings to OpenBLAS for matrix ops.

---

## Final Thoughts

This is a **genuinely hard problem**. The most likely failure mode is that attention weights end up being uninterpretable noise, even if the model achieves decent anomaly detection. The difference between a publishable result and a failed experiment will be:

1. **Rigorous ablations** showing attention is necessary
2. **Controlled synthetic experiments** where ground truth is known
3. **Human evaluation** of attention weights by domain experts
4. **Numerical correctness** verified at every layer

If you can show that attention weights genuinely point to causal events in controlled settings, and that this generalizes to real bugs, you'll have something worth presenting at a systems conference (OSDI, SOSP, ASPLOS).

**Expected timeline:** 3-4 months of focused work, assuming 20-30 hours/week.

**What would make this a strong research artifact:**

- Open-source release with reproducible results
- Benchmark dataset of annotated traces
- Comparison against symbolic debuggers (gdb scripts, rr)
- Case studies on real-world bugs (Linux kernel, databases)
