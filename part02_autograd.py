import torch

# Gradient Computation
## Compute gradient of scalar value
x = torch.randn(3, requires_grad=True)
# print(x)

y = x + 2
# print(y)
z = y * y * 2
z = z.mean()
# print(z)

z.backward()  # Calculate dz/dx, the gradient of z with respect of x
# print(x.grad)


## Compute gradient of array
x = torch.randn(3, requires_grad=True)
# print(x)

y = x + 2
# print(y)
z = y * y * 2
# print(z)

v = torch.tensor([1.0, 0.001, 0.1])
z.backward(v)
# print(x.grad)


# Untrack gradient
## Method#1
x = torch.ones(5, requires_grad=True)
x.requires_grad_(False)

## Method#2
x = torch.ones(5, requires_grad=True)
x.detach()

## Method#3
x = torch.ones(5, requires_grad=True)
with torch.no_grad():
    y = x + 2
    # print(y)
    ...


# Gradient accumulation
## backward() will accumulate the previous gradient
weights = torch.ones(4, requires_grad=True)

for epoch in range(2):
    model_output = (weights * 3).sum()

    model_output.backward()
    # print(weights.grad)

## use .zero_ to zero out the gradient
weights = torch.ones(4, requires_grad=True)

for epoch in range(2):
    model_output = (weights * 3).sum()

    model_output.backward()
    # print(weights.grad)
    weights.grad.zero_()

