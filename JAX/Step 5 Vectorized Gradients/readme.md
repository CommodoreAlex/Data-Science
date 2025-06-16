# Vectorized Gradients with `vmap`

## What You Learn

This example demonstrates how to combine **automatic differentiation** and **vectorization** in JAX using:

- `grad()` – to compute the derivative of a function
- `vmap()` – to apply a function over a batch of inputs without writing explicit loops

By using `vmap(grad(f))`, you efficiently compute gradients across a batch of inputs in parallel — a key feature in JAX's design for high-performance numerical computing.

---

## What the Code Does

We define a quadratic function:

```python
def f(x):
    return x ** 2 + 2 * x + 1
```

Then, we compute its derivative:
```python
f_grad = grad(f)  # Derivative: f'(x) = 2x + 2
```

To apply this derivative across many values at once, we vectorize it:
```python
batched_grad = vmap(f_grad)
```

This allows us to pass in an array of inputs and get their corresponding gradients, without explicit Python loops.

---

## Example Output

For the input array `[-2.0, 0.0, 1.0, 2.0]`:
```python
Function outputs: [1. 1. 4. 9.]
Gradient outputs: [-2. 2. 4. 6.]
```

Explanation:

- `f([-2, 0, 1, 2])` returns `[1, 1, 4, 9]`
    
- `f'([-2, 0, 1, 2])` returns `[-2, 2, 4, 6]`, which are the slopes at those points
    

---

## Why This Matters

This technique is important in:

- Batch gradient computation for machine learning models
    
- High-throughput scientific computations
    
- Writing readable code that performs well on CPU/GPU/TPU
    

You get the power of broadcasting and auto-differentiation, without writing for-loops or worrying about performance.

---
