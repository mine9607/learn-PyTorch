import numpy as np
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

np.set_printoptions(precision=3)

#### Creating Tensors ####

# Create Tensor from list
a = [1, 2, 3]
t_a = torch.tensor(a)
print(t_a)
print(t_a.shape)
# Create Tensor from numpy array
b = np.array([4, 5, 6], dtype=np.int32)
t_b = torch.from_numpy(b)
print(t_b)
print(tuple(t_b.shape))

# Create Tensor of shape (r,c) filled with 1.
t_ones = torch.ones(2, 3)
print(t_ones.shape)
print(t_ones)

# Create Tensor of random values - shape(r,c)
rand_tensor = torch.rand(2, 3)
print(rand_tensor)
print(rand_tensor.shape)
print(tuple(rand_tensor.shape))

#### Manipulating Tensors ####
"""
cast - use torch.to()
reshape - use 
transpose
squeeze
"""

# Change data type - torch.to()
t_a_new = t_a.to(torch.int64)
print(t_a_new.dtype)

# NOTE: certain operations require that input tensors have a certain number of dimensions (rank) associated with a certain number of elements (shape)

# Transposing a tensor
t = torch.rand(3, 5)
t_tr = torch.transpose(t, 0, 1)
print("Transpose: ", t.shape, " --> ", t_tr.shape)

# Reshaping a tensor
t = torch.zeros(30)
t_reshape = t.reshape(5, 6)
print("Reshape: ", t.shape, " --> ", t_reshape.shape)

# Squeezing a tensor
t = torch.zeros(1, 2, 1, 4, 1)
t_sqz = torch.squeeze(t, 2)
print("Squeeze: ", t.shape, " --> ", t_sqz.shape)

#### Applying Mathematical Operators ####


#### Split, Stack, Concatenate Tensors ####
