# Project 3: JAX Just-In-Time (JIT) Compilation

## Overview

This project demonstrates how to use **JAX's JIT compilation** to dramatically speed up mathematical computations. By wrapping a function with `jax.jit`, we instruct JAX to compile the function into optimized machine code the first time it is run, which can significantly accelerate repeated computations.

---

## Concepts Introduced

- **`jax.jit`**: JIT (Just-In-Time) compilation for transforming Python functions into XLA-optimized versions.
- **Compilation vs Execution**: First run compiles the function (slightly slower), future runs reuse the optimized version (much faster).
- **Lazy evaluation**: We use `.block_until_ready()` to ensure JAX has completed execution so we can accurately time it.

---

## Code Summary

We compute the **sum of squares** of a large vector using both:

1. A standard JAX function (`f`)
2. A JIT-compiled version (`f_jit`)

```python
from jax import jit
import jax.numpy as jnp

def f(x):
    return jnp.sum(x ** 2)

f_jit = jit(f)
```

We then compare execution times for each.

---
## Expected Output

The first call to `f_jit(x_val)` takes longer because it compiles the function. Subsequent calls are much faster:

```python
Normal function output: 3.3333282e+17, time: 0.130000 seconds
JIT function output: 3.3333282e+17, time: 0.200000 seconds
JIT function output second call: 3.3333282e+17, time: 0.005000 seconds
```

(The actual times will vary depending on your machine.)

---
## Why This Matters

- JIT allows Pythonic code to approach C/C++ performance.
    
- JIT is especially useful for heavy numerical computation in machine learning and scientific computing.
    
- JAX is designed around functional purity and static shapes, making it ideal for such optimizations.

---
