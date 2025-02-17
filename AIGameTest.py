import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# 调整后的模型定义
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# 初始化模型
model = MyModel()

# 加载权重文件
state_dict = torch.load('model.pth', map_location=torch.device('cpu'))

# 手动映射键
new_state_dict = {}
for key in state_dict.keys():
    if key.startswith('module.'):
        new_key = key[7:]  # 去掉 'module.' 前缀
    else:
        new_key = key
    new_state_dict[new_key] = state_dict[new_key]

# 加载新的 state_dict
try:
    model.load_state_dict(new_state_dict)
except RuntimeError as e:
    print(f"Error loading state_dict: {e}")
    print("Attempting to load with strict=False...")
    model.load_state_dict(new_state_dict, strict=False)

model.eval()  # 设置模型为评估模式

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载本地图像文件
def load_image(image_path):
    image = Image.open(image_path).convert('L')  # 转换为灰度图像
    image = transform(image)
    image = image.unsqueeze(0)  # 增加批次维度
    return image

# 预测函数
def predict(image_path):
    image = load_image(image_path)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

# 示例：预测本地图像文件
image_path = './dataset/test/images/frame_0.jpg'
prediction = predict(image_path)
print(f'Predicted class: {prediction}')
