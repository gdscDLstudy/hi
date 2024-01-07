import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import matplotlib.pyplot as plt

# 함수 사용법은 구글에 함수 그대로 검색하면 공식 사이트 나옵니다
# line 10, 13 : torchvision 라이브러리를 사용해서 mnist 데이터셋 다운
train = torchvision.datasets.MNIST('/MNIST', train=True, download=True,
                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Normalize((0.1307,), (0.3081,))]))
test = torchvision.datasets.MNIST('/MNIST', train=False, download=True,
                                  transform=transforms.Compose([transforms.ToTensor(),
                                                                transforms.Normalize((0.1307,), (0.3081,))]))

# line 19, 20 : 데이터셋을 DataLoader함수를 사용해서 for문으로 뽑기 좋게 만들기
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=train, batch_size=100, shuffle=False)

# 텐서 연산을 cpu나 gpu 둘 중 한 곳에서 다같이 해야해서 device를 정하는 부분, 저는 gpu사용해서 else cpu 부분이 없어요
cuda = torch.device('cuda')


# class MLP : torch nn.Module 클래스를 상속해서 만듬, nn.Module에는 forward, backward 등 다양한 기능들이 구현되있음. 자세한건 구글
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # layer를 nn.Sequential로 구현
        # nn.Sequential은 레이어들을 묶어서 만들기 좋음, 로우레벨 구현할 때는 안쓰게 될거에요
        # nn.Linear = fully-connected layer / 입력 사이즈가 28*28=784이니 (784, 256) 크기의 W만들어주는 겁니다. 여기서 256은 뉴런 갯수니까 알아서 정해보세요
        self.layer = nn.Sequential(nn.Linear(784, 256),
                                   nn.ReLU(True),
                                   nn.Linear(256, 128),
                                   nn.ReLU(True),
                                   nn.Linear(128, 10))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer(x)
        return x


#모델 객체 만들고
model = MLP()
#모델을 cuda(device)에 올려주는거
model = model.cuda()

# loss 정의
loss = nn.CrossEntropyLoss()
# opimizer 정의
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=5e-4, momentum=0.9)
cost = 0

# 학습 내용 기록용 리스트
iterations = []
train_losses = []
test_losses = []
train_acc = []
test_acc = []

# 10에폭만큼 학습
for epoch in range(10):
    # model.eval()하면 평가모드(evaluation)으로 바뀌고 기울기 갱신이 안됨. 그 후에 test_loader 가져와서 테스트
    model.eval()
    correct2 = 0
    for data, target in test_loader:
        data = data.to(cuda)
        target = target.to(cuda)
        output = model(data)
        cost2 = loss(output, target)
        prediction = output.data.max(1)[1]
        correct2 += prediction.eq(target.data).sum()

    # model.train()하면 train 모드가 되고 학습 가능
    model.train()
    correct = 0
    for X, Y in train_loader:
        X = X.to(cuda)
        Y = Y.to(cuda)
        optimizer.zero_grad()
        hypo = model(X)
        cost = loss(hypo, Y)
        cost.backward()
        optimizer.step()
        prediction = hypo.data.max(1)[1]
        correct += prediction.eq(Y.data).sum()

    # 학습 할 때 loss, accuracy 기록
    print("Epoch : {:>4} / cost : {:>.9}".format(epoch + 1, cost))
    iterations.append(epoch)
    train_losses.append(cost.tolist())
    test_losses.append(cost2.tolist())
    train_acc.append((100*correct/len(train_loader.dataset)).tolist())
    test_acc.append((100*correct2/len(test_loader.dataset)).tolist())


# 학습 끝나고 최종 정확도 한번 더 계산
model.eval()
correct = 0
for data, target in test_loader:
    data = data.to(cuda)
    target = target.to(cuda)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()

print('Test set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))

# matplotlib으로 학습 내용 그래프로 plot
plt.subplot(121)
plt.plot(range(1, len(iterations)+1), train_losses, 'b--')
plt.plot(range(1, len(iterations)+1), test_losses, 'r--')
plt.subplot(122)
plt.plot(range(1, len(iterations)+1), train_acc, 'b-')
plt.plot(range(1, len(iterations)+1), test_acc, 'r-')
plt.title('loss and accuracy')
plt.show()
