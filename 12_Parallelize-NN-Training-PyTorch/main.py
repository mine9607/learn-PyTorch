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
# Element-wise

torch.manual_seed(1)
t1 = 2 * torch.rand(5, 2) - 1
t2 = torch.normal(mean=0, std=1, size=(5, 2))
print(t1)
print(t2)

t3 = torch.multiply(t1, t2)
print(t3)

t4 = torch.mean(t1, axis=0)
print(t4)

t5 = torch.matmul(t1, torch.transpose(t2, 0, 1))
print(t5)
print(t5.shape)

t6 = torch.matmul(torch.t(t1), t2)
print(t6)
print(t6.shape)

norm_t1 = torch.linalg.norm(t1, ord=2, dim=1)
print(norm_t1)
print(norm_t1.shape)

#### Split, Stack, Concatenate Tensors ####

# Split a Tensor using torch.chunk()
"""
torch.chunk() - divides a tensor into a list of equally sized tensors
"""

print("\nSPLITTING")

# providing the number of splits
torch.manual_seed(1)
t = torch.rand(6)
print(t)

t_splits = torch.chunk(t, 3)
print([item.numpy() for item in t_splits])

# providing the sizes of different splits
t = torch.rand(5)
t_splits = torch.split(t, split_size_or_sections=[3, 2])
print(t_splits)
print([item.numpy() for item in t_splits])

# Stack (concatenate) Tensors using torch.stack() or torch.cat()

"""
torch.stack() - 
torch.cat()
"""
print("\nCONCATENATING")
A = torch.ones(3)
B = torch.zeros(2)
C = torch.cat([A, B], axis=0)

print(A)
print(B)
print(C)

print("\nSTACKING")
A = torch.ones(3)
B = torch.zeros(3)
S = torch.stack([A, B], axis=1)

print(A)
print(B)
print(S)
