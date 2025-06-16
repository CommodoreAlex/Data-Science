#!/usr/bin/env python3
import time
import jax.numpy as jnp
from jax import grad, jit   # Just-In-Time (JIT) Compilation

# Define a function that computes sum of squares of a large vector
def f(x):
    return jnp.sum(x ** 2)

# JIT-Compile the function for speed
f_jit = jit(f)

# Prepare a large vector of type float32
x_val = jnp.arange(1_000_000, dtype=jnp.float32)

# Time normal function
start = time.time()
y = f(x_val).block_until_ready()  # Wait for computation to finish
end = time.time()
print(f"Normal function output: {y}, time: {end - start:.6f} seconds")

# Time JIT-Compiled function on first call
start = time.time()
y_jit = f_jit(x_val).block_until_ready() # Wait for computation to finish
end = time.time()
print(f"JIT function output: {y_jit}, time: {end - start:.6f} seconds")

# Run again to see JIT speed-up on second call (compilation happens on first call)
start = time.time()
y_jit2 = f_jit(x_val).block_until_ready()
end = time.time()
print(f"JIT function output second call: {y_jit2}, time: {end - start:.6f} seconds")
