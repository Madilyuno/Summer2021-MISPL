'''
1.  Sigmoid Function : 1/1+e**(-x) -->sigmoid(x)
    Logistic Regression using Sigmoid Function --> H(x) = sigmoid(Wx+b)
    W의 값과 그래프의 경사도는 비례한다
    b의 값이 커질수록 그래프는 왼쪽으로 이동한다 (-b 만큼 그래프 평행이동)

2.  Cost Function : Linear Regression 에서 사용한 Cost Function 을 그대로 사용하면 안됨
    극값이 여러개 존재할 수 있음
    --> Gradient descent 가 Global Minimum 이 아닌 Local Minimum 에 도달할 수도 있기 때문
    Cost Function 새로 정의하여 Gradient Descent

# Forward 연산 : H(x) 식에 입력 x 를 대입하여 예측 값 y 를 얻는 연산
'''

# Sigmoid Function

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x) :
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y, 'g')
plt.plot([0,0],[1.0,0.0],':')
plt.title('Sigmoid Function')
plt.show()

# Multiple Logistic Regression

# Training
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# Train Data
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# Initialize Weight, Intercept
W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Set Optimizer as Gradient Descent
optimizer = optim.SGD([W, b], lr=1)

epochs = 1000

for epoch in range(epochs + 1) :
    # y = H(x) = xW + b  : Matrix Multiply x with W
    # Sigmoid Function
    # hypothesis = 1/ (1 + torch.exp(-(x_train.matmul(W) + b)))
    # H(x) 계산 : torch 에서 주어진 Sigmoid Function 사용
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)

    # Cost Function
    # cost = -(y_train * torch.log(hypothesis) + (1- y_train) * torch.log(1 - hypothesis)).mean()
    # Cost Function 계산 : torch 에서 주어진 Cost Function
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # Cost Function 으로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0 :
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, epochs, cost.item()))

# How to make Binary Classifier with Class

class BinaryClassifier(nn.Module) :
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = BinaryClassifier()

optimizer = optim.SGD(model.parameters(), lr=1)

epochs = 1000

for epoch in range(epochs + 1) :
    # y_hat = H(x) = xW + b  : Matrix Multiply x with W
    # Sigmoid Function
    # hypothesis = 1/ (1 + torch.exp(-(x_train.matmul(W) + b)))
    # H(x) 계산 : torch 에서 주어진 Sigmoid Function 사용
    hypothesis = model(x_train)

    # Cost Function
    # cost = -(y_train * torch.log(hypothesis) + (1- y_train) * torch.log(1 - hypothesis)).mean()
    # Cost Function 계산 : torch 에서 주어진 Cost Function
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # Cost Function 으로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 10 == 0 :
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}'.format(epoch, epochs, cost.item(), accuracy))