import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models.detection as detection_models
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载预训练的 Faster R-CNN 模型
model = detection_models.fasterrcnn_resnet50_fpn(pretrained=True)

# 修改模型以适应你的类别数
num_classes = 49  # 包括背景类
in_features = model.roi_heads.box_predictor.cls_score.in_features

# 替换 box_predictor 为与你的类别数兼容的结构
model.roi_heads.box_predictor = detection_models.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# 加载训练好的权重
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

# 加载本地图像文件
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  # ResNet需要RGB图像
    image_tensor = transform(image).unsqueeze(0)  # 增加批次维度
    return image_tensor, image

# 预测函数
def predict_detection(image_tensor, model, class_names):
    with torch.no_grad():
        predictions = model(image_tensor)

        # 解析预测结果
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        # 打印每个边界框及其类别
        detections = []
        for box, label, score in zip(boxes, labels, scores):
            if score > 0.5:  # 只显示置信度大于0.5的结果
                class_name = class_names[label]  # 目标检测模型的标签从1开始
                detections.append((box, class_name, score))

        return detections

def get_class_names_from_annotations(annotations_dir):
    class_names = ['__background__']  # 添加背景类

    # 遍历annotations目录下的所有XML文件
    for filename in os.listdir(annotations_dir):
        if filename.endswith('.xml'):
            file_path = os.path.join(annotations_dir, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()

            # 查找所有的<object>标签，并从中提取类别名称
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in class_names:
                    class_names.append(class_name)

    return class_names

# 绘制检测结果
def draw_detections(image, detections):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    for box, class_name, score in detections:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f'{class_name} {score:.2f}', color='white', backgroundcolor='red', fontsize=12)

    plt.axis('off')
    plt.show()

# 假设annotations目录路径
annotations_dir = './dataset/train/annotations'

# 获取类别名称
class_names = get_class_names_from_annotations(annotations_dir)

# 打印类别名称
print("Class names:", class_names)

# 示例：预测本地图像文件并处理多边界框
image_path = './dataset/test/images/frame_4230.jpg'
image_tensor, image = load_image(image_path)
detections = predict_detection(image_tensor, model, class_names)

print("Detections:")
for box, class_name, score in detections:
    print(f'Bounding Box: {box}, Predicted Class: {class_name}, Score: {score:.4f}')

# 绘制检测结果
draw_detections(image, detections)
