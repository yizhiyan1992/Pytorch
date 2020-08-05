import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,6,3)
        self.conv2=nn.Conv2d(6,16,3)
        self.fc1=nn.Linear(16*6*6,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        #max pooling over a (2,2) window
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(-1,self.num_flat_feature(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

    def num_flat_feature(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        return num_features

net=Net()

params=list(net.parameters())
print(params[0].size())
input=torch.randn(1,1,32,32)
out=net(input)
print(out)

#zero the gradient buffers of all parameters and backprops with random gradients:
#net.zero_grad()
#out.backward(torch.randn(1,10))

#computing the loss and updating the weights of the network
lossFunction=nn.MSELoss()
target=torch.randn(1,10)
loss=lossFunction(out,target)
print(loss)
print(loss.grad_fn)
#backprop
net.zero_grad()
loss.backward()
print(net.conv1.bias.grad)

import torch.optim as optim
optimizer=optim.SGD(net.parameters(),lr=0.01)
#in the training loop
optimizer.zero_grad()
output=net(input)
loss=lossFunction(output,target)
loss.backward()
optimizer.step()

x=torch.tensor([1.,2.,3.],requires_grad=True)
y=(x+2)**2
z=4*y
print(z)
