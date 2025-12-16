import torch
import torch.nn as nn
import torch.optim as optim #优化器
from torchvision import datasets, transforms
from torch.utils.data import DataLoader #数据加载器
from model import dnn

input_size=28*28
hidden_size1=128
hidden_size2=64
output_size=10

model=dnn(input_size,hidden_size1,hidden_size2,output_size)

criterion=nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

num_epochs=5
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0

    for images, labels in trainloader:
        images = images.view(images.size(0), -1)  # 根据实际 batch 大小扁平化

        # 清零优化器的梯度
        optimizer.zero_grad()

        # 向前传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新权重
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader)}")

# 评估模型
model.eval()  # 设置模型为评估模式
correct = 0
total = 0

with torch.no_grad():  # 在评估时不需要计算梯度
    for images, labels in testloader:
        images = images.view(images.size(0), -1)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # 获取预测的类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")

