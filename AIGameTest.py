import torch
import torch.nn as nn
import torch.nn.functional as F
# 在文件开头添加以下导入语句
from PIL import Image
import torchvision.transforms as transforms



# 定义一个简单的ResNet块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改输入通道数为3
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 修改输出类别数为48

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 初始化模型
model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=48)  # 修改输出类别数为48

# 加载权重文件
state_dict = torch.load('model.pth')

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
    transforms.Resize((224, 224)),  # ResNet通常需要224x224的输入
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载本地图像文件
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  # ResNet需要RGB图像
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
image_path = './dataset/test/images/frame_30.jpg'
prediction = predict(image_path)
print(f'Predicted class: {prediction}')
