import numpy as np
import torch
from transformers import is_tf_available


# Tensor creation
## Custom value and size tensor
x = torch.tensor([1, 2, 3])
# print(x)


## Normally distribute tensor of size (2, 2)
x = torch.rand(2, 2)
# print(x)


## One valued tensor of size (2, 2)
x = torch.ones(2, 2)
# print(x)


# Tensor operation
## Math operations
x = torch.tensor(1)
y = torch.tensor(2)
z = x + y
# print(z)

## Inline operations, apply transformation without assignment
## , can be invoke with additional "_" on method.
x = torch.tensor(1)
y = torch.tensor(2)
y.add_(x)
# print(y)


# Tensor indexing
## Basic row and col index
x = torch.rand(2, 2)
# print(x[0, 0])        # first row, first col
# print(x[0, :])        # first row
# print(x[:, 0])        # first col
# print(x[0, 0].item()) # get the actual value (only for 1 valued tensor)


# Tensor reshaping
## Basic reshaping
x = torch.rand(16)
# print(x.view(4, 4))     # reshape to (4, 4) tensor
# print(x.view(-1, 8))    # reshape to (16/8, 8) tensor (pytorch will determine the right size)


# Tensor size
## Find tensor size
x = torch.rand(3, 3)
# print(x.size())


# Tensor conversion to numpy
## Torch to Numpy
x = torch.tensor(1)
# print(x.numpy())

## Numpy to Torch
x = np.array([1, 2])
y = torch.from_numpy(x)
# print(y)

x += 1
# print(x)
# print(y) # x and y share the same memory address


# Tensor device
## Using cuda or cpu
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
x = torch.ones(5, device=device)
# print(x.device)

## Move to another device
y = torch.ones(5)
y.to(device)
# print(y.device)
