# Reproducibility (try to take random out of random) with random seed.
import torch
# random_tensor_A = torch.rand(3, 4)
# random_tensor_B = torch.rand(3, 4)

# print(random_tensor_A)
# print(random_tensor_B)
# print(random_tensor_A == random_tensor_B)

RANDOM_SEED = 4

torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)

print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)
