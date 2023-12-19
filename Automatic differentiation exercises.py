#!/usr/bin/env python
# coding: utf-8

# <h1><b>Exercises on Automatic differentiation</h1></b>

# 1. Why is the second derivative much more expensive to compute than the first derivative?

# The first derivative measures how a function's output changes concerning its input variables, 
# 
# while the second derivative represents the rate of change of the first derivative. 
# 
# The second derivative is the derivative of the derivative.

# In[ ]:


2. After running the function for backpropagation, immediately run it again and see what happens.


# Re-running the backpropagation function without any modifications, in most cases, 
# 
# will result in the same gradients and parameter updates, 
# 
# maintaining consistency in the optimization process.

# In[ ]:


3. In the control flow example where we calculate the derivative of d with respect to a, what
would happen if we changed the variable a to a random vector or matrix. At this point, the
result of the calculation f(a) is no longer a scalar. What happens to the result? How do we
get_ipython().run_line_magic('pinfo', 'this')


# Changing the variable aa from <i>a</i> scalar to a random vector or matrix alters the derivative calculation from a scalar to a vector or matrix output, respectively. This transforms the derivative <i>(d / da) d</i> into a gradient (for a vector output) or a Jacobian matrix (for a matrix output). Analyzing these higher-dimensional derivatives reveals how each element of the output changes concerning changes in each element of the input vector or matrix.
# 

# In[ ]:


4. Redesign an example of finding the gradient of the control flow. Run and analyze the result


# In[1]:


import torch

# Define a variable 'x' and initialize it with a value
x = torch.tensor(5.0, requires_grad=True)

# Define a computation that involves control flow
def computation(x):
    result = torch.tensor(0.0)
    for i in range(1, 5):
        if i % 2 == 0:  # Control flow based on even or odd 'i'
            result += torch.square(x)
        else:
            result += x
    return result

# Use PyTorch's autograd to compute gradient
y = computation(x)
gradient = torch.autograd.grad(y, x)[0]

# Run the computation and evaluate the gradient
result = y.item()
gradient_value = gradient.item()
print("Result of the computation:", result)
print("Gradient of the computation with respect to 'x':", gradient_value)


# 1. We created a tensor 'x' initialized with a value of 5 and set 'requires_grad=True' to track gradients.
# 2. The computation involves a loop with a control flow that checks if 'i' is odd or even and performs different operations based on this condition.
# 3. PyTorch's autograd.grad is used to compute the gradient of the output 'y' with respect to the variable 'x'.
# 4. Finally, we evaluate both the result of the computation and the computed gradient.

# 5. Let f(x) = sin(x). Plot f(x) and df(x)
# dx , where the latter is computed without exploiting that
# f′(x) = cos(x).

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x) = sin(x)
def f(x):
    return np.sin(x)

# Define the derivative estimation function using finite differences
def df_dx(x, h=0.001):
    return (f(x + h) - f(x - h)) / (2 * h)

# Generate x values
x_values = np.linspace(-2*np.pi, 2*np.pi, 1000)

# Calculate f(x) and df(x)/dx values
f_x = f(x_values)
df_dx_values = df_dx(x_values)

# Plot f(x) = sin(x)
plt.figure(figsize=(8, 6))
plt.plot(x_values, f_x, label='f(x) = sin(x)', color='blue')
plt.title('Plot of f(x) = sin(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

# Plot df(x)/dx using finite differences
plt.figure(figsize=(8, 6))
plt.plot(x_values, df_dx_values, label='df(x)/dx (Estimated)', color='red')
plt.title('Numerical Estimation of df(x)/dx')
plt.xlabel('x')
plt.ylabel('df(x)/dx')
plt.legend()
plt.grid(True)
plt.show()


# In[3]:


import torch
import matplotlib.pyplot as plt

# Define the function f(x) = sin(x)
def f(x):
    return torch.sin(x)

# Define the derivative estimation function using finite differences
def df_dx(x, h=0.001):
    return (f(x + h) - f(x - h)) / (2 * h)

# Generate x values
x_values = torch.linspace(-2 * torch.pi, 2 * torch.pi, 1000)

# Calculate f(x) and df(x)/dx values
f_x = f(x_values)
df_dx_values = df_dx(x_values)

# Plot f(x) = sin(x)
plt.figure(figsize=(8, 6))
plt.plot(x_values.numpy(), f_x.numpy(), label='f(x) = sin(x)', color='blue')
plt.title('Plot of f(x) = sin(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

# Plot df(x)/dx using finite differences
plt.figure(figsize=(8, 6))
plt.plot(x_values.numpy(), df_dx_values.numpy(), label='df(x)/dx (Estimated)', color='red')
plt.title('Numerical Estimation of df(x)/dx')
plt.xlabel('x')
plt.ylabel('df(x)/dx')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




