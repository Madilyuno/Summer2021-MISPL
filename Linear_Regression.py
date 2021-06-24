import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
'''
pytorch로 선형 회귀 구현
train 데이터 생성 --> tensor
가설 세우기 --> W, b (직선의 coefficient, intercept)
가설 --> cost function
gradient descent
'''

torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

'''
다중 선형 회귀일 경우 가중치 w를 각 피쳐의 개수만큼 지정하고 초기화해서 실행
matrix multiplication 으로 한번에 구현할 수 있다. (hypothesis = train_x.matmul(W) + b)
ex)
x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  80], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
'''
W = torch.zeros(1, requires_grad=True) # 자동 미분 == True 설정, backward() 함수 호출 시 자동으로 미분
b = torch.zeros(1, requires_grad=True)

epochs = 2000
for epoch in range(epochs + 1) :
    hypothesis = x_train * W + b
    cost = torch.mean((hypothesis - y_train)**2)  ##  평균제곱오차 MSE
    optimizer = optim.SGD([W,b], lr=0.01)
    optimizer.zero_grad() # 0으로 초기화, 없으면 미분값 누적
    cost.backward() # 비용함수를 미분하여 gradient 계산
    optimizer.step() # W,b 업데이트

    if epoch % 100 == 0 :
        print('Epoch {:4d}/{} W : {:.3f}, b : {:.3f}, Cost : {:.6f}'.format(epoch, epochs, W.item(), b.item(), cost.item()))

'''
Custom Dataset
'''
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                   [93, 88, 93],
                   [89, 91, 90],
                   [96, 98, 100],
                   [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

  # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size = 2, shuffle=True)

model = torch.nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)