'''
One-Hot Encoding : To solve Multi-Class Classification Problem
                   For Independent Classes that are Categorical Data
                   Make Vectors that have equivalent distance among them

Softmax Function : if the number of classes is K, input k-dim vectors and
                   calculate the probability of each class

Softmax Regression : Makes K dim Vectors (k is the number of classes)
                     Vectors get through Softmax Function
                     that makes the sum (of the result Vector's Component) is 1 (the sum of probability is one)

Hypothesis : H(X) = softmax(WX + b)

Vectors from features can be different from Input Vectors for Softmax Function
Then We have to do Dimension Reduction
We make anther Vector with Weight multiplication to feature vector (Weight must be updated to reduce error)
--> Think about the layer structure that first layer has nodes consist of feature and every edges have weights
    Next layer have nodes that multiplied with weights  --> That layer is Input Vectors for Softmax Function

Cost Function : Cross Entropy Function : (nll_loss function + (softmax function + log function))

If we do ().some_operation_() --> Inplace Operation
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

z = torch.FloatTensor([1, 2, 3])

hypothesis = F.softmax(z, dim=0)
hypothesis.sum()

z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)

y = torch.randint(5, (3,)).long()
y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train) # floating point tensor
y_train = torch.LongTensor(y_train) # integer tensor

y_one_hot = torch.zeros(8, 3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)

W = torch.zeros((4, 3), requires_grad=True) # 8 by 4 Matmul W = 8 by 3 --> W must be 4 by 3
b = torch.zeros(1, requires_grad=True)

for epoch in range(epochs + 1) :
    z = x_train.matmul(W) + b
    cost = F.cross_entropy(z, y_train)  # Cross Entropy Function consist of Cost Function and Softmax Function
                                        # The reason why we don't need to make seperated softmax function

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0 :
        print('Epoch {:4d}/{} Cost {:.6f}'.format(epoch, epochs, cost.item()))

class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,3)

    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()

optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 1000

for epoch in range(epochs + 1) :
    prediction = model(x_train)
    cost = F.cross_entropy(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0 :
        print('Epoch {:4d}/{} Cost {:.6f}'.format(epoch, epochs, cost.item()))






















