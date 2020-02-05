# Source: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html

import torch


# MARK: Operations
x = torch.ones(5, 3)
y = torch.ones(5, 3)
z = x + y
print(z)

y += x    # It works.
print(y)

y += 2    # It works! Every element got increased by 2.
print(y)


# MARK: Size
x = torch.ones(5, 3)
y = x[:, 1]
print(x.size())
print(x)
print(y.size())
print(y)


# MARK: Reshape
x = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8])
z1 = x.view(2, 4)
z2 = x.view(-1, 4)    # -1 means calculate automatically.

print(z1.size())
print(z1)
print(z2.size())
print(z2)


x = torch.Tensor(list(range(8)))
y = x.view(2, 1, 2, 2)
z = y.view(2, 2, 2)

print(y.size())
print(y)
print(z.size())
print(z)
