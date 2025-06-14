import torch

# Check for GPU access for PyTorch
print(torch.__version__)
print(torch.cuda.is_available())

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#Count number of devices
print(torch.cuda.device_count())

# Putting tensors (and models) on GPU
tensor_A = torch.tensor([1, 2, 3])
print(tensor_A, tensor_A.device)
tensor_on_gpu = tensor_A.to(device)
print(tensor_on_gpu)

# Move tensor back to CPU (to use numpy)
tensor_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_on_cpu)
