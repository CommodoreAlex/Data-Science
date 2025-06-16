#!/usr/bin/env python3
from jax import vmap
import jax.numpy as jnp

# Define a simple scalar function
def square(x):
    return x ** 2

# Create a vectorized version of the function using vmap
square_vectorized = vmap(square)

# Define an array of inputs
x_vals = jnp.array([1.0, 2.0, 3.0, 4.0])

# Apply both functions
scalar_outputs = [square(x) for x in x_vals]    # Manually applied
vectorized_outputs = square_vectorized(x_vals)  # Automatically batched

print(f"Manual application: {scalar_outputs} \n")
print("Vectorized with vmap:", vectorized_outputs)
