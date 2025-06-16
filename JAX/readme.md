# JAX Installation Guide for Windows (VS Code)

This guide provides a complete step-by-step process to install and run JAX in a Python virtual environment using Visual Studio Code on Windows.

---

## 1. Create a Python Virtual Environment

Open a terminal in VS Code (or Command Prompt) and run:

```cmd
python -m venv jaxenv
```

This creates a virtual environment named `jaxenv` in your current directory.

---

## 2. Activate the Virtual Environment

Still in the terminal, activate the virtual environment:
```cmd
jaxenv\Scripts\activate
```

You should now see your prompt change to:
```cmd
(jaxenv) C:\Users\yourname\YourProjectFolder>
```

This means you're working inside the virtual environment.

---

## 3. Install JAX (CPU Version)

Install JAX for CPU (recommended for development):
```cmd
pip install jax
```

This installs the core JAX library along with CPU-optimized dependencies.

---

## 4. Verify Installation

Check that JAX and its dependencies were installed:
```cmd
pip list
```

Expected output:
```cmd
Package     Version
----------- -------
jax         0.6.1
jaxlib      0.6.1
ml_dtypes   0.5.1
numpy       2.3.0
opt_einsum  3.4.0
pip         24.2
scipy       1.15.3
```

---

## 5. Configure Python Interpreter in VS Code

1. Press `Ctrl+Shift+P` in VS Code
    
2. Select: `Python: Select Interpreter`
    
3. Choose the interpreter ending in:
    

```cmd
jaxenv\Scripts\python.exe
```

If it’s not listed, click **Enter interpreter path** and browse to:
```cmd
<your_project_folder>\jaxenv\Scripts\python.exe
```

This ensures IntelliSense and code execution work inside your JAX environment.

---

## 6. Test the JAX Installation

Create a new file called `test_jax.py` and paste the following:
```python
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0])
print(x * 2)
```

Run the script in the terminal:
```cmd
python test_jax.py
```

Expected output:
```cmd
[2. 4. 6.]
```

---

## 7. Notes

This setup uses the CPU-only version of JAX. GPU acceleration on Windows requires using WSL2 with a compatible NVIDIA driver and CUDA toolkit.

To deactivate the virtual environment, simply run:

```cmd
deactivate
```

---
