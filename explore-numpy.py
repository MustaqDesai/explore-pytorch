import torch
import numpy as np

numpy_array = np.arange(1.0, 8.0)
tensor_from_num = torch.from_numpy(numpy_array)
print(numpy_array)
print(tensor_from_num)
numpy_from_tensor =tensor_from_num.numpy()
print(numpy_from_tensor)
print(numpy_from_tensor.dtype)






