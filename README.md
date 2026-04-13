# 🧠 Functional Neural Network in OCaml — Simplified Documentation

This project builds a neural network from scratch in **OCaml**, using a **pure functional approach**.

- No mutable variables  
- Uses recursion and higher-order functions  
- Focus is on **clarity + mathematical correctness**

---

# ⚙️ Core Module: `Nn`

The `Nn` module handles:
- Activation functions
- Their derivatives
- Loss functions

---

# 1️⃣ Functional Abstractions (`map` and `map2`)

Instead of loops, we use:

## 🔹 `map`
- Applies a function to every element in a matrix  
- Used for **activations**

👉 Example: Apply sigmoid to all values

---

## 🔹 `map2`
- Applies a function to corresponding elements of two matrices  
- Used for:
  - Loss computation
  - Gradients

---

## 💡 Why this is important

- Keeps code **clean and reusable**
- Separates:
  - *math logic* (what to compute)
  - *data traversal* (how to iterate)

---

# 2️⃣ Activation Functions

## 🔹 Sigmoid

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

### ✔ Purpose
- Converts values into range **(0, 1)**
- Useful for probabilities

---

### ✔ Derivative

$$
\sigma'(x) = y(1 - y)
$$

(where \( y = \sigma(x) \))

👉 Important:
- Uses **post-activation value**
- Avoids recomputing `exp`
- Makes backprop faster

---

## 🔹 ReLU (Rectified Linear Unit)

$$
f(x) = \max(0, x)
$$

### ✔ Purpose
- Keeps positive values
- Zeros out negatives
- Simple and fast

---

### ✔ Derivative

$$
f'(x) =
\begin{cases}
1 & x > 0 \\
0 & x \le 0
\end{cases}
$$

👉 Important:
- Uses **pre-activation value (x)**
- Needs original input before activation

---

# 3️⃣ Loss Functions

Loss tells us:

👉 *“How wrong is the model?”*

---

## 🔹 Mean Squared Error (MSE)

$$
\text{MSE} = \frac{1}{n} \sum (p - t)^2
$$

### ✔ Steps
1. Compute `(prediction - target)^2` using `map2`
2. Flatten matrix → list
3. Sum values
4. Divide by total elements \( n \)

---

### ✔ Gradient

$$
\frac{2}{n}(p - t)
$$

👉 Meaning:
- If prediction is high → decrease it  
- If low → increase it  

---

## 🔹 Binary Cross Entropy (BCE)

$$
L = -[t \log p + (1-t)\log(1-p)]
$$

### ✔ Purpose
- Used for **binary classification**
- Better than MSE for probabilities

---

### ⚠️ Numerical Stability (Very Important)

Before applying log:

$$
p = \text{clamp}(p, 10^{-15}, 1 - 10^{-15})
$$

👉 Prevents:
- `log(0)` → crash  
- NaN / infinity values  

---

### ✔ Gradient

$$
\frac{1}{n} \cdot \frac{p - t}{p(1 - p)}
$$

👉 Meaning:
- Confident wrong predictions → large updates  
- Correct predictions → small updates  

---

# 4️⃣ Generic Design

The implementation is **fully generic**:

- Uses `float list list`
- No fixed matrix size
- Works for:
  - Single input
  - Batch inputs

👉 Any dataset → as long as it can be converted to floats

---
