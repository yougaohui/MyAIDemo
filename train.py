import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from PIL import Image
import xml.etree.ElementTree as ET
from torch.nn.utils.rnn import pad_sequence



# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_folder, annotation_folder, transform=None):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.transform = transform
        self.image_files = os.listdir(image_folder)
        self.class_to_idx = self._get_class_to_idx()

    def _get_class_to_idx(self):
        class_to_idx = {}
        for img_name in self.image_files:
            annotation_path = os.path.join(self.annotation_folder, img_name.replace('.jpg', '.xml'))
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                label = obj.find('name').text
                if label not in class_to_idx:
                    class_to_idx[label] = len(class_to_idx)
        return class_to_idx

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        annotation_path = os.path.join(self.annotation_folder, img_name.replace('.jpg', '.xml'))

        image = Image.open(img_path).convert("RGB")
        boxes, labels = self.parse_annotation(annotation_path)

        if self.transform:
            image = self.transform(image)

        return image, boxes, labels

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            labels.append(self.class_to_idx[label])  # 使用 class_to_idx 转换为类别索引
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        return boxes, labels

# 自定义 collate_fn
def custom_collate_fn(batch):
    images, boxes, labels = zip(*batch)

    # 对 images 进行堆叠
    images = torch.stack(images, dim=0)

    # 对 boxes 进行填充
    boxes = [torch.tensor(box, dtype=torch.float32) for box in boxes]
    boxes = pad_sequence(boxes, batch_first=True, padding_value=-1)  # 使用 -1 进行填充

    # 对 labels 进行填充
    labels = [torch.tensor(label, dtype=torch.long) for label in labels]
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # 使用 -1 进行填充

    return images, boxes, labels

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整到更大的尺寸
    transforms.RandomCrop(224),  # 随机裁剪
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = CustomDataset('dataset/train/images', 'dataset/train/annotations', transform=transform)
test_dataset = CustomDataset('dataset/test/images', 'dataset/test/annotations', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

# 定义模型
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.class_to_idx))  # 根据实际类别数量定义输出层

# 添加Dropout层以防止过拟合
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, len(train_dataset.class_to_idx))
)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=-1)  # 忽略填充值 -1
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 使用Adam优化器

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 初始化TensorBoard
writer = SummaryWriter()

# 训练模型
num_epochs = 20  # 增加训练轮数
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, boxes, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        # 只使用第一个标签进行分类
        loss = criterion(outputs, labels[:, 0])  # 使用第一个标签进行分类
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # 每10个batch记录一次损失
        if i % 10 == 9:
            writer.add_scalar('Training Loss', running_loss / 10, epoch * len(train_loader) + i)
            running_loss = 0.0

    # 更新学习率
    scheduler.step()

    # 计算训练集准确率
    model.eval()
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for images, boxes, labels in train_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels[:, 0]).sum().item()

    train_accuracy = 100 * train_correct / train_total
    writer.add_scalar('Training Accuracy', train_accuracy, epoch)

    # 计算测试集准确率
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, boxes, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels[:, 0]).sum().item()

    test_accuracy = 100 * test_correct / test_total
    writer.add_scalar('Testing Accuracy', test_accuracy, epoch)

    print(f'Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader)}, Training Accuracy: {train_accuracy}%, Testing Accuracy: {test_accuracy}%')

writer.close()
print('Finished Training')

# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 模型评估
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, boxes, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels[:, 0]).sum().item()

print(f'Final Accuracy: {100 * correct / total}%')
