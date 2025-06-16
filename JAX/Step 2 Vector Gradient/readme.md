# Step 2: Automatic Differentiation with JAX - Vector Input

## Objective

In this step, you will extend your understanding of automatic differentiation by working with vector inputs.  

You will define a function \( f(\mathbf{x}) = \sum_i x_i^2 \) that takes a vector and computes the sum of squares, and then use JAX to compute the gradient with respect to the vector \(\mathbf{x}\).

---
## Why Do This Exercise?

- Gradients of vector functions are central to machine learning and scientific computing.  

- Understanding how JAX handles vector inputs shows the power of automatic differentiation in multidimensional contexts.  

- The gradient of a scalar function with vector input is a vector of partial derivatives, which JAX computes effortlessly.

---
## What You Should Know After This Exercise

- JAX’s `grad` computes gradients of functions with vector inputs and scalar outputs.  
- The output gradient is a vector of partial derivatives, one for each input dimension.  
- This generalizes the scalar case to multidimensional functions, which is essential for real-world tasks like training neural networks.

---

## Expected Output

```bash
f([1. 2. 3.]) = 14.0
f'([1. 2. 3.]) = [2. 4. 6.]
```

---
