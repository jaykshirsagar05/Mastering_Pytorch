# Imports
import torch
from torch._C import device
import torch.nn as nn
from torch.nn.modules import module
import torch.optim as optim
import torch.nn.functional as F #relu,tanh... functions with no paramaeters
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create FCN
class FCN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# model = FCN(784, 10)
# x = torch.randn(64, 784)  # O/p [64,10]
# print(model(x).shape)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load data
train_dataset = datasets.MNIST(root='/dataset', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='/dataset', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = FCN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
for epoch in range(num_epochs):
    for batch_idx ,(data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        data = data.reshape(data.shape[0],-1) #flatten reshaping for our defined inputs
        # print(data.shape)

        #forward
        outs = model(data)
        loss = criterion(outs, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient descent
        optimizer.step() #weights updation

# Testing

def check_acc(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0],-1)

            outs = model(x)
            _, predicts = outs.max(1)
            num_correct += (predicts == y).sum()
            num_samples += predicts.size(0)

        print(f'accuracy of {loader.dataset} is {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()
check_acc(train_loader, model)
check_acc(test_loader, model)
