from modal import App, Image
from modal.volume import Volume

app = App("product-classifier")
output_volume = Volume.from_name("product-classifier-output-vol")
data_volume = Volume.from_name("product-classifier-data-vol")

image = (
    Image.debian_slim()
    .pip_install([
        "torch", "torchvision", "Pillow", "scikit-learn", "tensorboard"
    ])
    .run_commands(
        "mkdir -p /data /output"
    )
    .add_local_python_source("datalore", "sitecustomize")
)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3600,
    volumes={
        "/output": output_volume,
        "/data": data_volume
    }
)
def train():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision import transforms, models
    from torchvision.models import ResNet50_Weights
    from PIL import Image
    import json
    from pathlib import Path
    import logging
    from sklearn.model_selection import train_test_split
    from torch.utils.tensorboard import SummaryWriter
    from collections import Counter
    from torch.amp import GradScaler, autocast
    import time
    import os

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Обновленные пути к данным
    DATA_ROOT = Path("/data")
    DATASET_ROOT = DATA_ROOT / "root" / "dataset"
    IMAGES_ROOT = DATASET_ROOT / "raw_images"
    LABELS_PATH = DATASET_ROOT / "labels_reorganized.json"
    OUTPUT_ROOT = Path("/output")

    def collect_image_data():
        """Собирает данные о изображениях и их метках"""
        image_paths = []
        labels = []

        logger.info(f"Loading labels from: {LABELS_PATH}")

        try:
            with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                labels_data = json.load(f)
                logger.info(f"Loaded {len(labels_data)} entries from labels file")
        except Exception as e:
            logger.error(f"Error loading labels file: {e}")
            raise

        # Проверяем существование директории с изображениями
        if not IMAGES_ROOT.exists():
            raise FileNotFoundError(f"Images directory not found: {IMAGES_ROOT}")

        logger.info(f"Scanning images in: {IMAGES_ROOT}")

        # Собираем изображения согласно labels_reorganized.json
        for json_path, label in labels_data.items():
            # Извлекаем только relevant parts пути
            # Из "dataset\\raw_images\\10006\\0.jpg" получаем ["10006", "0.jpg"]
            path_parts = json_path.split('\\')
            product_id = path_parts[-2]  # Получаем ID продукта
            image_name = path_parts[-1]  # Получаем имя файла

            # Собираем правильный путь
            abs_path = IMAGES_ROOT / product_id / image_name

            if abs_path.exists():
                image_paths.append(abs_path)
                labels.append(label)
                if len(image_paths) % 1000 == 0:
                    logger.info(f"Processed {len(image_paths)} images...")
            else:
                logger.warning(f"Image not found: {abs_path}")

        logger.info(f"Found {len(image_paths)} valid images")
        logger.info(f"Found {len(set(labels))} unique product categories")

        if not image_paths:
            logger.error("No valid images found!")
            logger.error("Sample paths tried:")
            for i, (path, label) in enumerate(zip(image_paths[:5], labels[:5])):
                logger.error(f"{i}: {path} -> {label}")
            raise ValueError("Empty dataset - no valid images found")

        return image_paths, labels

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
                return None

    class ProductClassifier(nn.Module):
        def __init__(self, num_classes):
            super(ProductClassifier, self).__init__()
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
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
        label_to_count = Counter(labels)
        weights = [1.0 / label_to_count[label] for label in labels]
        sampler = WeightedRandomSampler(weights, len(weights))
        return sampler

    def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, writer):
        best_val_acc = 0.0
        patience = 5
        patience_counter = 0
        scaler = GradScaler()
        accumulation_steps = 2

        # Создаем директорию для сохранения если её нет
        OUTPUT_DIR = Path("/output")
        OUTPUT_DIR.mkdir(exist_ok=True)

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            optimizer.zero_grad()
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if inputs is None:
                    logger.warning("Skipping batch due to None inputs (image loading error).")
                    continue
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                loss = loss / accumulation_steps
                scaler.scale(loss).backward()
                running_loss += loss.item() * accumulation_steps
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            if (batch_idx + 1) % accumulation_steps != 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            class_correct = torch.zeros(len(train_loader.dataset.category_to_idx))
            class_total = torch.zeros(len(train_loader.dataset.category_to_idx))
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.float16):
                    for inputs, labels in val_loader:
                        if inputs is None:
                            logger.warning("Skipping validation batch due to None inputs (image loading error).")
                            continue
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                        for label, pred in zip(labels, predicted):
                            class_correct[label] += (label == pred).item()
                            class_total[label] += 1
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * correct / total
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            samples_per_sec = len(train_loader.dataset) / epoch_duration
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Speed/epoch_duration', epoch_duration, epoch)
            writer.add_scalar('Speed/samples_per_sec', samples_per_sec, epoch)
            for idx, (correct, total) in enumerate(zip(class_correct, class_total)):
                if total > 0:
                    class_acc = 100. * correct / total
                    writer.add_scalar(f'Accuracy/class_{train_loader.dataset.idx_to_category[idx]}',
                                      class_acc, epoch)
            scheduler.step(val_loss)
            logger.info(f'Epoch {epoch + 1}/{num_epochs}:')
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            logger.info(f'Epoch Duration: {epoch_duration:.2f} seconds, Samples/sec: {samples_per_sec:.2f}')
            if val_acc > best_val_acc:
                logger.info(f"Validation accuracy improved from {best_val_acc:.2f} to {val_acc:.2f}")
                best_val_acc = val_acc

                # Сохраняем модель
                model_save_path = OUTPUT_DIR / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'category_mapping': train_loader.dataset.idx_to_category
                }, str(model_save_path))
                logger.info(f"Model saved to {model_save_path}")

                # Сохраняем маппинг категорий
                mapping_save_path = OUTPUT_DIR / "category_mapping.json"
                with open(mapping_save_path, 'w') as f:
                    json.dump(train_loader.dataset.idx_to_category, f, indent=2, ensure_ascii=False)
                logger.info(f"Category mapping saved to {mapping_save_path}")

                patience_counter = 0
            else:
                patience_counter += 1

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return None, None
        return torch.utils.data.dataloader.default_collate(batch)

    def main():
        BATCH_SIZE = 64
        NUM_EPOCHS = 50
        LEARNING_RATE = 0.001
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Проверяем структуру каталогов
        logger.info("Checking directory structure...")
        logger.info(f"DATA_ROOT exists: {DATA_ROOT.exists()}")
        logger.info(f"DATASET_ROOT exists: {DATASET_ROOT.exists()}")
        logger.info(f"IMAGES_ROOT exists: {IMAGES_ROOT.exists()}")
        logger.info(f"LABELS_PATH exists: {LABELS_PATH.exists()}")

        # Выводим содержимое каталогов для отладки
        logger.info("Directory contents:")
        try:
            logger.info(f"IMAGES_ROOT contents:")
            for item in IMAGES_ROOT.iterdir():
                if item.is_dir():
                    sample_files = list(item.iterdir())[:2]
                    logger.info(f"  {item.name}/: {[f.name for f in sample_files]}")
        except Exception as e:
            logger.error(f"Error listing directory contents: {e}")

        # Загружаем данные
        image_paths, labels = collect_image_data()

        logger.info(f"Successfully loaded {len(image_paths)} images with {len(set(labels))} categories")

        # Выводим примеры для проверки
        logger.info("Sample data:")
        for i in range(min(5, len(image_paths))):
            logger.info(f"Image {i}: {image_paths[i]} -> {labels[i]}")

        # Train/test split
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

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )

        num_classes = len(train_dataset.category_to_idx)
        model = ProductClassifier(num_classes=num_classes).to(DEVICE)

        if torch.cuda.device_count() > 1:
            logger.info(f"Используем {torch.cuda.device_count()} GPU!")
            model = nn.DataParallel(model)
        model.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        writer = SummaryWriter('/root/output/runs/product_classifier')

        with open(OUTPUT_ROOT / 'category_mapping.json', 'w') as f:
            json.dump(train_dataset.idx_to_category, f, indent=2)

        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, DEVICE, NUM_EPOCHS, writer)

        logger.info("Training completed!")

    main()



@app.local_entrypoint()
def run():
    train.remote()

if __name__ == "__main__":
    with app.run():
        run()
