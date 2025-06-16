#!/usr/bin/env python3
import jax.numpy as jnp # JAX's version of NumPy, supports automatic differentation and JIT
from jax import grad    # 'grad' function from JAX, which computes gradients (deriatives)

# Define a simple Python function: f(x) = x squared
# JAX needs functions to be written in pure Python using jax.numpy for compatibility
def f(x):
    return x ** 2   # This is the mathematical function: f(x) = x²

# Use JAX to create a new function that computes the gradient (i.e., derivative) of f(x)
# grad(f) returns a function f' such that f'(x) = d(f)/dx
f_grad = grad(f)

x_val = 3.0            # Test the function and its gradient at x = 3.0
y = f(x_val)           # Compute the function value
dy_dx = f_grad(x_val)  # Compute the derivative at x = 3.0

# Return the results
print(f"f({x_val}) = {y}")       # Expect 9.0, since 3^2 = 9
print(f"f'({x_val}) = {dy_dx}")  # Expect 6.0, since d(x²)/dx = 2x → 2×3 = 6

