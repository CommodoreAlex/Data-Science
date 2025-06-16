# Project 4: Vectorization with `jax.vmap`

## Overview

This project introduces `jax.vmap`, which is used to **automatically vectorize** functions over a batch dimension. This is especially useful for applying a function to many inputs without writing explicit loops.

---

## Key Concept

- **`vmap`** transforms a function written for **single inputs** into a function that works on **arrays of inputs** in a batched, efficient manner.
- This avoids Python loops and leverages JAX’s performance on GPU/TPU or CPU.

---

## Code Summary

We define a simple function `square(x)` and then use:

- A manual loop: `[square(x) for x in x_vals]`
- A vectorized call: `square_vectorized(x_vals)`

```python
def square(x):
    return x ** 2

square_vectorized = vmap(square)
```

---
## Output

```python
Manual application: [DeviceArray(1., dtype=float32), ..., DeviceArray(16., dtype=float32)]

Vectorized with vmap: [ 1.  4.  9. 16.]
```

(Note: Outputs might be displayed as `DeviceArray` depending on your setup.)

---
## Why This Matters

- Eliminates the need for manual batching logic.
    
- Works with any function that operates on individual elements (e.g., points, samples, rows).
    
- Under the hood, `vmap` is optimized and can be combined with `jit`.

---
