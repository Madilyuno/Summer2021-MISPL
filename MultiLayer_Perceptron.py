'''
Evaluation of Model

Hyper Parameter : Parameters that directly affect the Model (change by User)
    ex) learning rate, the number of hidden layers, neurons, rate of dropout

Parameter : Parameters that changes through training (change by the Model)
    ex) weights, bias

Use Validation data to tune the Hyper Parameters

Precision : True Positive / Answer Positive Data (True Positive + False Positive)

Recall : True Positive / Originally Positive Data (True Positive + False Negative)

Activation Function : An activation function defines how the weighted sum of the input is transformed
                      into an output from a node or nodes in a layer of the Network. (Especially Hidden Layer)
                      They are Nonlinear Functions. To accumulate hidden layers.
    ex) Sigmoid Function, Softmax Function, ReLU

In Neural Network, We use (original input) * weight as input of activation function. It continues Forward Propagation.
'''

# Single Layer Perceptron

'''

import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
if device == 'cuda' :
    torch.cuda.manual_seed_all(0)

X = torch.FloatTensor([[0,0], [0,1], [1,0], [1,1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

linear = nn.Linear(2, 1)
sigmoid = nn.Sigmoid()
model = nn.Sequential(linear, sigmoid).to(device)

criterion = nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

for step in range(10001) :
    optimizer.zero_grad()
    hypothesis = model(X)

    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if step % 100 == 0 :
        print(step, cost.item())
        
'''
# cost 가 200번 에포크 이후에는 줄어들지 않는다. --> 문제를 해결하지 못한다.

import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(0)
if device == 'cuda' :
    torch.cuda.manual_seed_all(0)

X = torch.FloatTensor([[0,0], [0,1], [1,0], [1,1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

model = nn.Sequential(nn.Linear(2,10),

                      nn.Sigmoid(), nn.Linear(10,10),

                      nn.Sigmoid(), nn.Linear(10,10),

                      nn.Sigmoid(), nn.Linear(10,1),
                      nn.Sigmoid()).to(device)

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

for epoch in range(10001) :
    optimizer.zero_grad()

    hypothesis = model(X)

    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0 :
        print(epoch, cost.item())

with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    print('실제값(Y): ', Y.cpu().numpy())
    print('정확도(Accuracy): ', accuracy.item())