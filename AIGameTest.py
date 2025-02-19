import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# 加载训练好的模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = SimpleCNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()


def predict(image, model):
    image_resized = cv2.resize(image, (128, 128))
    image_resized = image_resized / 255.0
    image_tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor).numpy().flatten()

    return prediction


# 实时检测
cap = cv2.VideoCapture('game.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    prediction = predict(frame, model)
    x1, y1, x2, y2 = prediction.astype(int)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('Game Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
