import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# 假设我们有一个模型定义
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28*28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 加载测试数据
test_dataset = MNIST(root='data', train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型
model = MyModel()
model.load_state_dict(torch.load('./model.pth'))  # 假设模型权重已经保存在model.pth文件中
model.eval()  # 设置模型为评估模式

# 测试模型
def test_model(model, test_loader):
    model.eval()  # 确保模型处于评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 不需要计算梯度
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')

# 运行测试
test_model(model, test_loader)
