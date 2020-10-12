import torch


"""
Math operations on tensors
"""

x = torch.tensor([1, 2, 3])
y = torch.tensor([3, 5, 7])

# addition
z1 = torch.add(x, y)
z2 = x+y


# subtraction
z = x - y

# division
z = torch.true_divide(x, y)

# inplace operations ("operation-name" + "_" ) <-- syntax
t = torch.zeros(3)
t.add_(x)
t += x

# Exponentiation
z = x ** 2

# comparison
z = x < 0

# matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)

# matrix exponentiation
matrix_exp = torch.rand(5, 5)
# print(matrix_exp.matrix_power(3))

# element wise
z = x * y
# print(z)
# dot product
# print(torch.dot(x, y))

# batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30
t1 = torch.tensor((batch, n, m))
t2 = torch.tensor((batch, m, p))
# out_bmm = torch.bmm()
# print(out_bmm)

# broadcasting in python and numpy

# useful operations
torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
# min, argmax, argmin, mean, eq(equal), any, all

"""
Indexing
"""

"""
Reshaping
"""

x = torch.arange(9)
x_3x3 = x.view(3,3)
x_3x3 = x.reshape(3,3)
print(x_3x3)