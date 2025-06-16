![image](https://github.com/user-attachments/assets/c09b4b96-8b33-4892-8c40-502f85de0a46)

## What is JAX?

[JAX](https://github.com/google/jax) is a high-performance numerical computing library developed by Google. It combines the ease of use of NumPy with the power of automatic differentiation and hardware acceleration (CPU, GPU, TPU).

It is designed for users who need:

- **Automatic differentiation**: Compute gradients, Jacobians, and Hessians of Python functions, useful in machine learning, optimization, and scientific computing.
- **Performance scaling**: JAX can compile and optimize code with `jit` and supports vectorization (`vmap`) and parallelization (`pmap`) across devices.
- **NumPy compatibility**: Most of the NumPy API is supported via `jax.numpy` with nearly identical syntax.
- **Functional programming**: JAX promotes pure functions and immutable data structures, enabling better composability and transformation.

Explore the official documentation here: [https://jax.readthedocs.io](https://jax.readthedocs.io)

---

## Why JAX?

JAX allows you to write concise mathematical code that:

- Runs efficiently on modern hardware (CPUs, GPUs, TPUs)
- Is easy to differentiate
- Can be automatically vectorized or parallelized
- Scales to complex workflows involving deep learning, scientific computing, and simulation

Its primary use cases include:

- Research and prototyping in machine learning
- Numerical methods and scientific simulations
- Any domain requiring performant and differentiable computation

---

## Setting Up JAX in VSCode

To get started with JAX in your development environment, refer to the setup instructions:

[Setup Guide for VSCode and JAX](vscode.md)

This guide walks through:

- Creating an isolated environment
- Configuring VSCode to work smoothly with the JAX ecosystem
- Running and troubleshooting basic scripts

---

This repository is organized as a progressive learning path, with each folder containing a small project demonstrating one or more JAX features. Continue with the individual projects to explore JAX concepts interactively.
