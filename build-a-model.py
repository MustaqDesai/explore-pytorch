import torch
from torch import nn
import matplotlib.pyplot as plt

# Create linear regression model class
class LinearRegressionModel(nn.Module): # Almost everyting from PyTorch inherits form nn.Module
    def __init__(self):
        super().__init__()
        # Initialize model parameters
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

        # Forward method to define the computation in the model
        def forward(self, x: torch.Tensor) -> torch.Tensor: # "x" is the input data
            return self.weight * x + self.bias # this is the linear regression formula
        
# Check the contents of PyTorch model using .parameters()

# Create a random seed
torch.manual_seed(42)

# Create an instance of the above model
model_0 = LinearRegressionModel()

# List the parameters
print(list(model_0.parameters()))

# List named parameters
print(model_0.state_dict())

# Make prediction using "torch.inference_mode()"
with torch.inference_mode():
    y_preds = model_0("")


        
