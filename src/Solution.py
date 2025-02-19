import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProductDataset(Dataset):
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


class ProductClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ProductClassifier, self).__init__()

        # 50 вместо 18 тк он получше
        self.backbone = models.resnet50(pretrained=True)

        # замораживаем начальные слои
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def create_weighted_sampler(labels):
    # балансер
    label_to_count = Counter(labels)
    weights = [1.0 / label_to_count[label] for label in labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, writer):
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

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

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        #валидация
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        class_correct = torch.zeros(len(train_loader.dataset.category_to_idx))
        class_total = torch.zeros(len(train_loader.dataset.category_to_idx))

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # точность по классам
                for label, pred in zip(labels, predicted):
                    class_correct[label] += (label == pred).item()
                    class_total[label] += 1

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total

        # метрики
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        for idx, (correct, total) in enumerate(zip(class_correct, class_total)):
            if total > 0:
                class_acc = 100. * correct / total
                writer.add_scalar(f'Accuracy/class_{train_loader.dataset.idx_to_category[idx]}',
                                  class_acc, epoch)

        scheduler.step(val_loss)

        logger.info(f'Epoch {epoch + 1}/{num_epochs}:')
        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        #early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'category_mapping': train_loader.dataset.idx_to_category
            }, 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f'Early stopping triggered after epoch {epoch + 1}')
            break


def main():
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f'Using device: {DEVICE}')

    with open('dataset/raw_images/labels_reorganized.json', 'r', encoding='utf-8') as f:
        labels_data = json.load(f)

    image_paths = [Path(path) for path in labels_data.keys()]
    labels = list(labels_data.values())

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ProductDataset(train_paths, train_labels, train_transform)
    val_dataset = ProductDataset(val_paths, val_labels, val_transform)

    train_sampler = create_weighted_sampler([train_dataset.category_to_idx[label] for label in train_labels])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    num_classes = len(train_dataset.category_to_idx)
    model = ProductClassifier(num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    writer = SummaryWriter('runs/product_classifier')

    with open('category_mapping.json', 'w') as f:
        json.dump(train_dataset.idx_to_category, f, indent=2)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, DEVICE, NUM_EPOCHS, writer)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()