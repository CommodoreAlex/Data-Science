# Step 1: Automatic Differentiation with JAX - Simple Scalar Function

---
## Objective

This step introduces you to one of JAX’s core features: **automatic differentiation (AD)**.  
You will define a simple mathematical function \( f(x) = x^2 \) and use JAX to automatically compute its derivative \( f'(x) = 2x \).

---
## Why Do This Exercise?

- **Understand the power of automatic differentiation:**  
  Normally, computing derivatives requires calculus and manual effort. JAX automates this for you accurately and efficiently.

- **Recognize why this is important:**  
  Derivatives are fundamental in many scientific and machine learning tasks such as training neural networks and solving optimization problems.

- **See how JAX simplifies your workflow:**  
  You write the function once, and JAX provides its gradient automatically—no manual derivative code needed.

---
## What You Should Know After This Exercise

- JAX’s `grad` function computes derivatives of scalar-output functions automatically.  
- Your function must be written as a **pure function** using `jax.numpy` operations.  
- This opens doors to advanced applications like optimization, sensitivity analysis, and machine learning.

---
## Expected Output

```bash
f(3.0) = 9.0
f'(3.0) = 6.0
```

---
