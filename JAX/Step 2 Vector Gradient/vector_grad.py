#!/usr/bin/env python3
import jax.numpy as jnp # JAX's version of NumPy, supports automatic differentation and JIT
from jax import grad    # 'grad' function from JAX, which computes gradients (deriatives)

# Define a function of a vector: sum of squares
def f(x):
    return jnp.sum(x ** 2)

# Create a gradient function that returns the gradient vector of f
f_grad = grad(f)

# Test vector input
x_val = jnp.array([1.0, 2.0, 3.0])

# Compute function value
y = f(x_val)

# Compute gradient value
dy_dx = f_grad(x_val)

# Return results
print(f"f({x_val}) = {y}")
print(f"f'({x_val}) = {dy_dx}")
