import math
import torch


x = torch.tensor([[1, 2, 3], [3, 2, 1], [2, 1, 2]], dtype=torch.float32)
y = torch.tensor([2, 3, 1], dtype=torch.float32)
w = torch.rand(x.shape[-1], requires_grad=True)
print(x, y, w)


max_error = 1e-3
learning_rate = 1e-4
loss = math.inf

while loss > max_error:
    # Linear Regression
    z = torch.matmul(x, w)

    # Compute mse and its grad
    loss = torch.sqrt((y - z)**2)
    loss = loss.mean()

    # Gradient descent
    loss.backward()
    with torch.no_grad():
        w += learning_rate * -w.grad
        w.grad.zero_()

    print(f"Loss: {loss}")