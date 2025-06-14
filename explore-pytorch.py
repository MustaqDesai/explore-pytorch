import torch
# print(torch.__version__)

# scalar
# scalar = torch.tensor(7)
# print(scalar)
# print(scalar.ndim)
# Get tensor back as Python int
# print(scalar.item())

# vector
# vector = torch.tensor([1,2])
# print(vector)
# print(vector.ndim)
# print(vector[0])

# MATRIX
# MATRIX = torch.tensor([[3,4],[5,6]])
# print(MATRIX)
# print(MATRIX.ndim)
# print(MATRIX[0])
# print(MATRIX.size())

# TENSOR
# TENSOR = torch.tensor([[[1,2,3],[4,4,6],[7,8,9]]])
# print(TENSOR)
# print(TENSOR.ndim)
# print(TENSOR[0])
# print(TENSOR.size())
# print(TENSOR.shape)

# Random tensors
# Create a random tensor of size (3,4)
# random_tensor = torch.rand(5,1)
# print(random_tensor)
# print(random_tensor.ndim)
# print(random_tensor.shape)

# Create a random tensor with similar shape to an image tensor
# random_image_size_tensor = torch.rand(size=(224,224,3)) # height, width, colour channels (R, G, B)
# print(random_image_size_tensor.ndim)
# print(random_image_size_tensor.shape)

# Zeros and ones
# zeros = torch.zeros(3,4)
# print(zeros)

# ones = torch.ones(3,4)
# print(ones)
# print(ones.dtype)

# Creating a range of tensors and tensors-like
# Use torch.arange()
# one_to_ten = torch.arange(1,11)
# print(one_to_ten)

# Creating tensors like
# one_to_ten_like = torch.zeros_like(one_to_ten)
# print(one_to_ten_like)

# Tensor datatype
# fload 32 tensor
# float_32_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=None)
# print(float_32_tensor)
# print(float_32_tensor.device)
# print(float_32_tensor.dtype)
# print(float_32_tensor.shape)

# Manipulating Tensors (tensor operations)
# addition, subtraction, multiplicaiton, division, matrix multiplication

# tensor = torch.tensor([1,2,3])
# print(tensor + 10)
# print(tensor * 10)
# print(tensor - 10)

# Element wise multiplication
# print(tensor, "*", tensor)
# print(f"Equals: {tensor * tensor}")

# Matrix multiplication (dot product)
# tensor([1, 2, 3]) * tensor([1, 2, 3])
# 1*1 + 2*2 + 3*3
# print(torch.matmul(tensor,tensor))

# for loop 
# value = 0
# for i in range(len(tensor)):
#     value += tensor[i] * tensor[i]

# print(value)

# Shapes for matrix multiplication
# tensor_AB = torch.tensor([[1, 2],
#                           [3, 4],
#                           [5, 6]])

# tensor_BA = torch.tensor([[7, 10],
#                           [8, 11],
#                           [9, 12]])

# torch.mm is same as torch.matmul
# torch.mm(tensor_AB, tensor_BA)

# To fix out tensor issues, manipulate the same of one tensors using tranpose
# print(tensor_BA.T)
# print(torch.mm(tensor_AB, tensor_BA.T))
# print(torch.mm(tensor_AB, tensor_BA.T).shape)

# Tensor aggregation
# Create a tensor
tensor_x = torch.arange(0, 100, 10)
# print(tensor_x)
# print(tensor_x.min())
# print(tensor_x.max())
# print(tensor_x.sum())
# print(tensor_x.mean()) # RuntimeError: mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Long
# print(torch.mean(tensor_x.type(torch.float32)))
# print(tensor_x.type(torch.float32).mean())

# Finding the position of min and max
# print(tensor_x.argmin())
# print(tensor_x.argmax())

# Reshaping - reshape an input tensor to a defined shape
tensor_y = torch.arange(1., 10.)
# print(tensor_y)
# print(tensor_y.shape) 
tensor_y_reshaped = tensor_y.reshape(1, 9)
# print(tensor_y_reshaped)
# View - return a view of an input tensor of certain shape, but keep the same memory as the original tensor
# tensor_z = tensor_y.view(1, 9)
# print(tensor_z)
# print(tensor_z.shape)

# changing tensor_z will change tensor_y
# tensor_z[:,0] = 5
# print(tensor_y)
# print(tensor_z)
# Stacking - combine multiple tensors on top of each other
# tensor_stacked = torch.stack([tensor_x, tensor_x, tensor_x, tensor_x],dim=0)
# print(tensor_stacked)

# Squeezing - removes size 1 tensors
# print(tensor_y_reshaped)
# print(tensor_y_reshaped.shape)
# tensor_sqeezed = tensor_y_reshaped.squeeze()
# print(tensor_sqeezed)
# print(tensor_sqeezed.shape)

# Unsqeezhing - adds a single dimention at a specific dimention
# tensor_unsqueezed = tensor_sqeezed.unsqueeze(dim=0)
# print(tensor_unsqueezed)
# print(tensor_unsqueezed.shape)

# Permute - re-arranges the dimensions in a speicfied order
# tensor_original = torch.rand(224,224, 3) # height, width, color channle
# print(tensor_original.shape)
# tensor_permuted = tensor_original.permute(2, 0, 1) # 3rd dim (index 2), then 1st dim (index 0), then 2nd dim (index 1)
# print(tensor_permuted.shape)

# Indexing (selecting data from tensors)
# tensor_data = torch.arange(1,10).reshape(1,3,3)
# print(tensor_data)
# print(tensor_data.shape)

# print(tensor_data[0])
# print(tensor_data[0][0])
# print(tensor_data[0][0][0])
# print(tensor_data[0][1][1])
# print(tensor_data[0][2][2])

# Get all of a dimention
# print(tensor_data[:,0])


