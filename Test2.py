import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 生成数据集
torch.manual_seed(0)
X = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = X + 0.3 * torch.rand(X.size())


# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegressionModel()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 评估模型
model.eval()
predicted = model(X).detach()

# 绘制结果
plt.plot(X.numpy(), y.numpy(), 'ro', label='Original data')
plt.plot(X.numpy(), predicted.numpy(), label='Fitted line')
plt.legend()
plt.show()
