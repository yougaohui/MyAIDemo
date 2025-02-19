import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import xml.etree.ElementTree as ET
from train import CustomDataset, custom_collate_fn
import torch.nn as nn

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整到更大的尺寸
    transforms.CenterCrop(224),  # 中心裁剪
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
test_dataset = CustomDataset('dataset/test/images', 'dataset/test/annotations', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

# 定义模型
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, len(test_dataset.class_to_idx))
)

# 加载训练好的模型权重
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 模型评估
correct = 0
total = 0
with torch.no_grad():
    for images, boxes, labels in test_loader:
        images = images.to(device)  # 将图像数据移动到设备
        labels = labels.to(device)  # 将标签数据移动到设备
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels[:, 0]).sum().item()

print(f'Final Accuracy: {100 * correct / total}%')

# 单个图像推理示例
def predict_single_image(image_path, model, transform, class_to_idx):
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    # 应用变换
    image_tensor = transform(image).unsqueeze(0).to(device)
    # 进行推理
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    # 获取预测类别
    predicted_class = list(class_to_idx.keys())[list(class_to_idx.values()).index(predicted.item())]
    return predicted_class

# 示例图像路径
image_path = './dataset/test/images/frame_0.jpg'
predicted_class = predict_single_image(image_path, model, transform, test_dataset.class_to_idx)
print(f'Predicted class for {image_path}: {predicted_class}')
