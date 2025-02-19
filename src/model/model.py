import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ToyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        unique_categories = sorted(set(labels))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            label = self.category_to_idx[self.labels[idx]]
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return self.__getitem__(0)


class ToyClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(ToyClassifier, self).__init__()

        # ResNet18
        self.backbone = models.resnet18(pretrained=True)

        # веса backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # замена последнего слоя на классификационную головку
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, writer):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Валидация
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total

        # Логируем метрики
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        logger.info(f'Epoch {epoch + 1}/{num_epochs}:')
        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')


def main():
    # Конфигурация
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загрузка разметки
    with open('dataset/raw_images/labels.json', 'r') as f:
        labels_data = json.load(f)

    # Подготовка данных
    image_paths = [Path(path) for path in labels_data.keys()]
    labels = list(labels_data.values())

    # Разделение на train и validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Аугментации для тренировочного сета
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Преобразования для валидационного сета
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Создание датасетов
    train_dataset = ToyDataset(train_paths, train_labels, train_transform)
    val_dataset = ToyDataset(val_paths, val_labels, val_transform)

    # Создание даталоадеров
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Создание модели
    num_classes = len(train_dataset.category_to_idx)
    model = ToyClassifier(num_classes=num_classes).to(DEVICE)

    # Определение функции потерь и оптимизатора
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Создание writer для TensorBoard
    writer = SummaryWriter('runs/toy_classifier')

    # Сохраняем маппинг категорий
    with open('category_mapping.json', 'w') as f:
        json.dump(train_dataset.idx_to_category, f, indent=2)

    # Обучение модели
    train_model(model, train_loader, val_loader, criterion, optimizer, DEVICE, NUM_EPOCHS, writer)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()