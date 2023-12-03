#!/usr/bin/env python
# coding: utf-8

# <h1><b1></i>Data manipulation</h1></b1></i>

# In[1]:


import torch


# In[2]:


torch.zeros((2, 3, 4))


# In[3]:


torch.ones((2, 3, 4))


# In[4]:


torch.randn(3, 4)

"""
Prints a tensor of shape (3, 4)
Each element in the tensor is sampled from a standard Guassian (normal) 
distribution with a mean of 0 and a standard deviation of 1
"""


# In[5]:


"""
We can also specify the exact values of the individual element
We apply the python lists containing numerical values
"""
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])


# Common arithmetic operations in PyTorch include:
# 
# <b>Addition (+)</b>
# 
# <b>Subtraction (-) </b>
# 
# <b>Multiplication (*)</b>
# 
# <b>Division (/)</b>
# 
# <b>Exponentiation (**)</b>

# Let's do a series of data manipulation tasks.

# In[6]:


a = torch.tensor([1.0, 2, 4, 8])
b = torch.tensor([2, 2, 2, 2])

a + b # Addition


# In[7]:


a - b # Subtraction


# In[8]:


a / b # Division


# In[9]:


a ** b # Exponentiation

"""
In mathematics, exponentiation is raising a number to its power.
For example, the power of 1 is 1.
When you raise 8 to power 8, you get 64. 
"""


# In[10]:


torch.exp(a)

"""
This is unary-like exponentiation
"""


# We can perform <b>linear algebra</b> operations, including;
#     
# <b>Vector dot productions</b>
#     
# <b>Matrix manipulation</b>

# <h2><b>Concatenation of tensors</h2></b>
# 
# Concatenation of tensors stack them together from end-to-end.
# 
# This forms a large tensor

# In[15]:


q = torch. arange(12, dtype=torch.float32).reshape((3, 4))
r = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((q, r), dim=0), torch.cat((q, r), dim=1)


# In[ ]:




