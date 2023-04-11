import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class LeNet5(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNet5, self).__init__()

        # Определяем слои свертки и пулинга
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # 6 ядер свертки размера 5x5
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # максимальное значение в окне 2x2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 16 ядер свертки размера 5x5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # максимальное значение в окне 2x2

        # Определяем полносвязные слои
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 120 нейронов в полносвязном слое
        self.fc2 = nn.Linear(120, 84)  # 84 нейрона в полносвязном слое
        self.fc3 = nn.Linear(84, num_classes)  # выходной слой с числом нейронов, соответствующим числу классов

    def forward(self, x):
        # Пропускаем входное изображение через слои свертки и пулинга
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Выпрямляем выход из второго слоя пулинга и пропускаем через полносвязные слои
        x = x.view(-1, 16 * 5 * 5)  # размерность выхода из второго слоя пулинга (16, 5, 5)
                                    # изменяем на (размер пакета, 16*5*5) для подачи на вход полносвязному слою
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    # Задаем размер входного изображения
input_size = 224

# Определение гиперпараметров
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Создание экземпляра модели LeNet-5
model = LeNet5()

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Создание экземпляра обучающей выборки
train_transforms = transforms.Compose([
    transforms.Resize((32, 32)), # изменение размера изображения до 32x32 пикселей
    transforms.ToTensor(), # преобразование изображения в тензор PyTorch
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # нормализация значений пикселей изображения
])
train_dataset = FaceDataset(data_dir='lfw', split='train', transform=train_transforms)

# Создание экземпляра загрузчика данных для обучающей выборки
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Цикл обучения
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Прямой проход
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Обратное распространение и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Вывод статистики
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

print('Обучение завершено!')

# Заменяем последний слой (классификатор) на новый слой, который будет распознавать 5 классов
num_classes = 5
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Задаем функцию потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Загружаем данные для обучения и валидации
data_dir = 'path/to/data'
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

# Создаем загрузчики данных
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)


# загрузка модели
model = LeNet5()

# загрузка тестовой выборки
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# вычисление accuracy на тестовой выборке
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_true += targets.cpu().numpy().tolist()
        y_pred += predicted.cpu().numpy().tolist()

accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)
# загрузка модели
model = torch.load('model.pt')
model.eval()

# функция для применения модели к новому изображению
def predict(image_path):
    # загрузка изображения и применение трансформаций
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image).unsqueeze(0)
    
    # применение модели к изображению и получение предсказания
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1)
        
    # возвращение предсказания
    return prediction.item()

# создание пользовательского интерфейса
from tkinter import *
from tkinter import filedialog

def open_image():
    # загрузка изображения и получение предсказания
    file_path = filedialog.askopenfilename()
    prediction = predict(file_path)
    
    # отображение результата
    if prediction == 0:
        result_label.config(text="Это лицо")
    else:
        result_label.config(text="Это НЕ лицо")

# создание окна с кнопкой для загрузки изображения и местом для отображения результата
window = Tk()
window.title("Распознавание лиц")
window.geometry("300x200")

open_button = Button(window, text="Открыть изображение", command=open_image)
open_button.pack(pady=20)

result_label = Label(window, text="")
result_label.pack()

window.mainloop