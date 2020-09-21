import torch
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.utils.data as Data
import cv2
DOWNLOAD_MNIST = False
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=50, shuffle=True)
#plt.imshow(train_data.train_data[10].numpy(),cmap='gray')
#plt.show()
#print(train_data.train_data[1])
test_data = torchvision.datasets.MNIST(root='./mnist/',train=False)

test_x = torch.unsqueeze(test_data.data,dim=1).type(torch.FloatTensor)[:2000]/255
test_y = test_data.targets[:2000]

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.out = torch.nn.Linear(32*7*7,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output
cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(),lr=0.001)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(1):
    for step, (b_x, b_y) in enumerate(train_loader):   # 分配 batch data, normalize x when iterate train_loader
        print(b_x.shape)
        output = cnn(b_x)   
        print(b_y)            # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # a

test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')



for t in range(1):
    b = 0
    for i in train_data.data[:10]:
        i = torch.unsqueeze(i,dim=0)
        i = torch.unsqueeze(i,dim=0)
        i = i.float()
        i = i/255
        output = cnn(i)
        print(output)
        print(torch.unsqueeze(train_data.targets[b],0))
        loss = loss_func(output,torch.unsqueeze(train_data.targets[b],0))
        print(train_data.targets[b])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        b = b+1
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')

torch.save(cnn,'net.pkl')
torch.save(cnn.state_dict(),'net_parameters.pkl')
