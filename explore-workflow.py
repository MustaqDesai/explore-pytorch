import torch
from torch import nn
import matplotlib.pyplot as plt



# Check PyTorch version
# print(torch.__version__)

# Create some *known* data using the linear regression forula.
# Create known paramters
weight = 0.7
bias = 0.3

# Print weight and bias
print(f"Weight: {weight}, Bias: {bias}")

# Create
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# Split data into training and test sets
# Create a train/test split
train_set = int (0.8 * len(X))
# print(train_split)
X_train, y_train = X[:train_set], y[:train_set]
X_test, y_test = X[train_set:], y[train_set:]

# print(len(X_train))
# print(len(y_train))
# print(len(X_test))
# print(len(y_test))

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    Plots tranining data, test data, and compares predictions.
    """
    # plt.figure(figsize=(10, 7))

    # Plot trianing data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")

     # Plot test data in blue
    plt.scatter(test_data, test_labels, c="g", s=4, label="Test Data")

    # Are tehre predictions?
    if predictions is not None:
        # Plot the predictions if they exist
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})
    
    plt.show() # has an option parameter.

plot_predictions()

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
# print(list(model_0.parameters()))

# List named parameters before training
print(model_0.state_dict())

# Make prediction using "torch.inference_mode()"
with torch.inference_mode():
    y_preds = model_0(X_test)

# print(y_preds)

# Predicitons before training
plot_predictions(predictions=y_preds)

# Train model to move from unknown prameters to known parameters

# Setup a loss function
loss_fn = nn.L1Loss()

# Setup an optimizer (stochastc gradiant descent)
optimizer = torch.optim.SGD(model_0.parameters(),lr=0.01) # lr = learning rate

# Training loop
# Loop through the data, forward pass/propagation to make predictions on data, calculate the loss ...
# Calculate the loss, optimizer zero grad, loss backward, optimizer step
# One loop through the data ...
epochs = 100
loss = 0.0
# Pass the data throught he model for a number of ephochs (e.g. 100)
for epoch in range(epochs):
    # set the model to training mode
    model_0.train() # train mode sets all parameters that require dradiants to require gradiants.
    # Forward pass/propagation on the training data
    y_preds = model_0(X_train)
    # Calculate the loss
    loss = loss_fn(y_preds, y_train)
    
    # Zero the graidants of the optimizer (default is to accumulate) 
    optimizer.zero_grad()

    # Perform back propagation on the loss with respect to the parameters of the model
    loss.backward()
    
    # Step the optimzer (perform gradiant descent)
    optimizer.step()

    # Testing
    model_0.eval()


# List loss 
print(f"loss: {loss}")

# List named parameters after training
print(model_0.state_dict())

# Predictions after training
with torch.inference_mode():
    y_preds_new = model_0(X_test)
   
plot_predictions(predictions=y_preds_new)



        

    