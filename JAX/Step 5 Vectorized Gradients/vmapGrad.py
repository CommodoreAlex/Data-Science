#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from jax import grad, vmap

# Define a scalar function
def f(x):
    return x ** 2 + 2 * x + 1

# Compute its gradient
f_grad = grad(f)

# Now vectorize the gradient across many x values
batched_grad = vmap(f_grad)

# Inputs
x_vals = jnp.array([-2.0, 0.0, 1.0, 2.0])
y_vals = f(x_vals)
grad_vals = batched_grad(x_vals)

print("Function outputs:", y_vals)
print("Gradient outputs:", grad_vals)
