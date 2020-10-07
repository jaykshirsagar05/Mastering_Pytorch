import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device)

print(my_tensor.device)

x = torch.empty(size=(3, 3))
print(x)
x = torch.zeros((2, 2))
print(x)
x = torch.arange(1, 5, 2)
print(x)
# initialize and conversion

my_tensor = torch.arange(4)
print(my_tensor.bool())

# numpy array <-> tensor
import numpy as np

np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
print(tensor)
np_array_back = tensor.numpy()
print(np_array_back)