# 🧠 Functional Neural Network from Scratch in OCaml

A complete neural network implementation built **from scratch** in OCaml using a **pure functional approach** — no mutable state, no iterative loops, no external ML libraries.

**Course:** Programming Languages  
**Team:** Prateek Rath (IMT2022068), Ketan Ghungralekar (IMT2022058)

---

## ✨ Key Highlights

- **Pure Functional:** No `ref`, no `for`/`while` loops — only recursion, `map`, `fold`, and `map2`
- **From Scratch:** Matrix library, activations, losses, forward pass, backpropagation, training loop — all hand-written
- **Real-World Results:** Trained on 45,000 loan samples → **89.08% accuracy**

---

## 📁 Project Structure

```
NN_ocaml/
├── matrix.ml          # Pure functional matrix library
├── matrix.mli         # Matrix module interface
├── nn.ml              # Neural network library (activations, losses, forward/backward, training)
├── nn.mli             # Neural network module interface
├── train_loan.ml      # Loan default prediction — CSV parsing, preprocessing, training
├── test_matrix.ml     # Unit tests for matrix operations
├── test_nn.ml         # Unit tests for NN primitives
├── loan_data.csv      # Kaggle loan default dataset
├── Makefile           # Build system
└── report/
    └── mid_eval_report.tex  # Mid-evaluation report (LaTeX)
```

---

## ⚙️ Module Architecture

### Matrix (`matrix.ml`)

Pure functional matrix operations using `float list list`:

| Operation | Function |
|-----------|----------|
| Transpose | `transpose` |
| Dot product | `dot_product` |
| Matrix multiply | `matmul` |
| Element-wise ops | `map`, `map2`, `add`, `sub`, `mul_elementwise` |
| Scalar multiply | `mul_scalar` |
| Axis reductions | `sum_axis_0`, `max_axis_0`, `min_axis_0` |
| Utilities | `make`, `flatten`, `shape` |

### Neural Network (`nn.ml`)

| Component | Functions |
|-----------|-----------|
| Activation functions | `sigmoid`, `relu`, `tanh`, `softmax` |
| Derivatives | `sigmoid_derivative`, `relu_derivative`, `tanh_derivative` |
| Loss functions | `mse`, `bce`, `cross_entropy` |
| Loss gradients | `mse_derivative`, `bce_derivative`, `cross_entropy_derivative` |
| Training | `forward`, `backward`, `update_model`, `train` |

---

## 🔢 Algorithms Implemented

### Forward Pass
Recursively traverses the layer list, computing $Z = XW + b$ and applying activation (ReLU for hidden layers, Softmax for output).

### Backpropagation
Reverse traversal computing gradients via the chain rule. Uses the softmax + cross-entropy shortcut: $\delta_L = (\hat{Y} - Y) / m$.

### Mini-Batch Gradient Descent
Data split into batches → forward pass → backward pass → weight update per batch, repeated for configurable epochs.

---

## 🚀 Building & Running

### Prerequisites
- OCaml compiler (`ocamlc`)

### Build
```bash
make all       # Build everything
make clean     # Remove compiled files
make re        # Clean + rebuild
```

### Run
```bash
./test_matrix          # Run matrix tests
./test_nn              # Run NN tests
./train_loan           # Train on loan dataset
```

---

## 📊 Results — Loan Default Prediction

| Parameter | Value |
|-----------|-------|
| Dataset | Loan Default (Kaggle) |
| Samples | 45,000 |
| Features | 13 (age, income, employment, credit history, etc.) |
| Architecture | 13 → 16 (ReLU) → 2 (Softmax) |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 32 |
| Learning Rate | 0.01 |
| Epochs | 20 |

### Training Output
```
Loading dataset (full)...
Loaded 45000 samples.
Starting training...
Epoch 20, Loss: 0.311287
Epoch 19, Loss: 0.278459
...
Epoch 2, Loss: 0.231218
Epoch 1, Loss: 0.231184
Final model loss: 0.231184
Accuracy: 89.08%
Time taken: 4.65 seconds
```

---

## 🔬 Functional Programming Concepts Used

| Concept | Usage |
|---------|-------|
| **Pure functions** | All functions are side-effect free (except I/O for CSV reading) |
| **Recursion** | Replaces all loops — forward pass, backward pass, training, data chunking |
| **Higher-order functions** | `map`, `map2`, `fold_left` for matrix operations and element-wise computation |
| **Pattern matching** | Layer traversal, CSV parsing, feature encoding |
| **Algebraic data types** | `layer` record type, `model` as layer list |
| **Module interfaces** | `.mli` files for encapsulation — `layer` type is abstract externally |
| **Immutable data** | All weight updates create new matrices, no in-place mutation |

---

## 📄 Report

The mid-evaluation report is available at [`report/mid_eval_report.tex`](report/mid_eval_report.tex). Compile with `pdflatex` or upload to [Overleaf](https://www.overleaf.com).

---

## 📚 References

1. Michael Nielsen, *Neural Networks and Deep Learning*, 2015 — [neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com/chap2.html)
2. Minsky, Madhavapeddy, Hickey, *Real World OCaml*, 2nd ed., 2022 — [dev.realworldocaml.org](https://dev.realworldocaml.org/)
3. Leroy et al., *The OCaml System: Documentation*, INRIA, 2024 — [ocaml.org](https://v2.ocaml.org/api/)
4. Goodfellow, Bengio, Courville, *Deep Learning*, MIT Press, 2016
