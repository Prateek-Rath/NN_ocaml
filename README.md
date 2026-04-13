# Functional Neural Network in OCaml - Detailed Documentation

This project implements a fully functional neural network library from scratch in OCaml, following a pure functional paradigm. It avoids all mutable state, using higher-order functions and recursion for clean, predictable logic.

## Core Module: `Nn`

The `Nn` module handles activations, derivatives, and loss functions. Below is a deep dive into the implementation details.

---

### 1. Functional Abstractions (`map` and `map2`)
To maintain a fully functional approach, we avoid iterative loops. Instead, we use higher-order functions from the `Matrix` module:

- **`map`**: Used for all activation functions. It applies a function to every individual float in the matrix.
- **`map2`**: Used for loss gradients and subtraction. It applies a binary function to elements at the same position in two different matrices.

By abstracting the list recursion into `map` and `map2`, the logic in `nn.ml` remains focused on the math rather than the data traversal.

---

### 2. Activation Functions & Their Logic

#### Sigmoid
- **Implementation**: `let sigmoid_fn x = 1.0 /. (1.0 +. exp (-.x))`
- **Derivative**: `s * (1.0 - s)`
- **Detail**: We apply the derivative to the **post-activation** value. If you have already calculated `y = sigmoid(x)`, the gradient is simply `y * (1 - y)`. This saves us from re-calculating the exponential function during backpropagation.

#### ReLU (Rectified Linear Unit)
- **Implementation**: `if x > 0.0 then x else 0.0`
- **Derivative**: `if x > 0.0 then 1.0 else 0.0`
- **Detail**: Unlike Sigmoid, the ReLU derivative is calculated based on the **pre-activation** input (`x`). If the input was positive, it passes the gradient through (1.0); otherwise, it blocks it (0.0).

---

### 3. Loss Functions & Scaling

The loss functions return a single `float` representing the average error across the entire batch/dataset.

#### Mean Squared Error (MSE)
- **Logic**: 
  1. Calculate `(prediction - target)^2` for every element using `map2`.
  2. Use `flatten` to turn the matrix into a single list of floats.
  3. Use `List.fold_left` to sum the list.
  4. Divide by $n$ (total number of elements) to get the average.
- **Gradient**: `(2.0 / n) * (prediction - target)`. The `1/n` scaling ensures that gradients don't explode as the dataset size increases.

#### Binary Cross Entropy (BCE)
- **Logic**: Uses the formula `- [t * log(p) + (1-t) * log(1-p)]`.
- **Numerical Stability**: We implement **Epsilon Shielding**. Before taking the `log`, we constrain the prediction `p` to be within `[1e-15, 1 - 1e-15]`. This prevents the code from attempting to calculate `log(0)`, which would return `NaN` or `-Infinity` and crash the training process.
- **Gradient**: `(1/n) * (p - t) / (p * (1 - p))`. This provides the direction to update weights to minimize the classification error.

---

### 4. Genericity
The library is "Generic" because it does not assume any specific shape for the matrices (rows or columns). 
- It works for a single sample (vector) as well as a large batch (matrix).
- It relies solely on the structure of `float list list`, making it compatible with any dataset that can be converted to floats.

## Summary of Logic Flow
1. **Forward Pass**: Data enters -> `Matrix.matmul` -> `Nn.activation`.
2. **Loss Calculation**: `Nn.mse` or `Nn.bce` calculates total error.
3. **Backward Pass**: `Nn.loss_derivative` -> `Nn.activation_derivative` -> update weights.
