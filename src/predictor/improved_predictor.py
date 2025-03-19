import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path
import logging
import os
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AlbumentationsDataset(Dataset):
    """Dataset with Albumentations transforms for more advanced augmentation"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        unique_categories = sorted(set(labels))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}

        # For error recovery
        self.backup_image_idx = 0
        while self.backup_image_idx < len(self.image_paths):
            try:
                Image.open(self.image_paths[self.backup_image_idx]).convert('RGB')
                break
            except:
                self.backup_image_idx += 1

        if self.backup_image_idx >= len(self.image_paths):
            self.backup_image_idx = 0
            logger.error("No valid backup image found!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            # Use PIL to open image
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)

            if self.transform:
                try:
                    # Apply albumentations transformations
                    transformed = self.transform(image=image_np)
                    image_tensor = transformed["image"]
                except Exception as e:
                    logger.error(f"Error during transformation of {image_path}: {e}")
                    # Fallback to backup image
                    backup_path = self.image_paths[self.backup_image_idx]
                    backup_image = np.array(Image.open(backup_path).convert('RGB'))
                    transformed = self.transform(image=backup_image)
                    image_tensor = transformed["image"]
            else:
                # Convert manually if no transform
                image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1).astype(np.float32) / 255.0)

            label = self.category_to_idx[self.labels[idx]]
            return image_tensor, label

        except Exception as e:
            logger.error(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return backup image
            return self.__getitem__(self.backup_image_idx)


class AdvancedProductClassifier(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b0', pretrained=True, dropout_rate=0.3):
        super(AdvancedProductClassifier, self).__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )

        # If using EfficientNet, we can customize the classifier
        if 'efficientnet' in model_name:
            # Get the classifier and replace it with our custom version
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        return self.model(x)


def create_advanced_transforms():
    """Create train and validation transforms using Albumentations"""
    train_transform = A.Compose([
        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0)),
        A.Rotate(limit=15),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),  # Only for some specific products
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(p=1),
            A.MotionBlur(p=1),
        ], p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=1),
            A.GridDistortion(p=1),
            A.ElasticTransform(p=1),
        ], p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return train_transform, val_transform


def create_weighted_sampler(labels):
    """Create a weighted sampler to balance classes"""
    label_to_count = Counter(labels)
    # Calculate weights inversely proportional to class frequency
    weights = [1.0 / label_to_count[label] for label in labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler


def validate_images(image_paths, num_workers=4):
    """Validate all images before training to avoid runtime errors"""
    logger.info("Validating images...")
    valid_paths = []
    invalid_paths = []

    def check_image(path):
        try:
            img = Image.open(path).convert('RGB')
            img.verify()  # Verify it's a valid image
            return True
        except Exception as e:
            logger.error(f"Invalid image {path}: {e}")
            return False

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(check_image, image_paths),
            total=len(image_paths),
            desc="Validating images"
        ))

    valid_paths = [path for path, is_valid in zip(image_paths, results) if is_valid]
    invalid_paths = [path for path, is_valid in zip(image_paths, results) if not is_valid]

    logger.info(f"Found {len(valid_paths)} valid images and {len(invalid_paths)} invalid images")
    return valid_paths, invalid_paths


def train_with_kfold(model_class, image_paths, labels, num_folds=5, batch_size=32,
                     num_epochs=30, learning_rate=1e-3, device='cuda', model_name='efficientnet_b0'):
    """Train with k-fold cross-validation for better generalization"""

    # Create a DataFrame for easier handling
    df = pd.DataFrame({
        'image_path': [str(p) for p in image_paths],
        'label': labels
    })

    # Convert labels to categorical indices
    unique_categories = sorted(set(labels))
    category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
    idx_to_category = {idx: cat for cat, idx in category_to_idx.items()}
    df['label_idx'] = df['label'].map(category_to_idx)

    # Initialize KFold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Prepare transforms
    train_transform, val_transform = create_advanced_transforms()

    # Track metrics across folds
    fold_val_accuracies = []

    # For each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(df['image_path'], df['label_idx'])):
        logger.info(f"Starting fold {fold + 1}/{num_folds}")

        # Split data
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # Create datasets
        train_dataset = AlbumentationsDataset(
            [Path(p) for p in train_df['image_path']],
            train_df['label'].tolist(),
            transform=train_transform
        )

        val_dataset = AlbumentationsDataset(
            [Path(p) for p in val_df['image_path']],
            val_df['label'].tolist(),
            transform=val_transform
        )

        # Create sampler for training data
        train_sampler = create_weighted_sampler(train_df['label_idx'].tolist())

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Initialize models
        num_classes = len(category_to_idx)
        model = model_class(num_classes=num_classes, model_name=model_name).to(device)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        # Create TensorBoard writer
        writer = SummaryWriter(f'runs/product_classifier_fold_{fold}')

        # Train models
        best_val_acc = 0.0
        patience_counter = 0
        max_patience = 7

        for epoch in range(num_epochs):
            # Training
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for inputs, labels in train_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar
                train_bar.set_postfix({
                    'loss': running_loss / (train_bar.n + 1),
                    'acc': 100. * correct / total
                })

            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            # Track per-class metrics
            class_correct = torch.zeros(num_classes).to(device)
            class_total = torch.zeros(num_classes).to(device)

            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc="Validation"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    # Per-class accuracy
                    for label, pred in zip(labels, predicted):
                        class_correct[label] += (label == pred).item()
                        class_total[label] += 1

            val_loss = val_loss / len(val_loader)
            val_acc = 100. * correct / total

            # Update learning rate
            scheduler.step(val_loss)

            # Record metrics
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

            for idx in range(num_classes):
                if class_total[idx] > 0:
                    class_acc = 100. * class_correct[idx] / class_total[idx]
                    writer.add_scalar(f'Accuracy/class_{idx_to_category[idx]}', class_acc, epoch)

            logger.info(f'Fold {fold + 1} - Epoch {epoch + 1}/{num_epochs}:')
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # Check for improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                # Save best models for this fold
                checkpoint_path = f'model_fold_{fold}_best.pth'
                torch.save({
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'category_mapping': idx_to_category
                }, checkpoint_path)

                logger.info(f"Saved best models for fold {fold + 1} with accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Record fold accuracy
        fold_val_accuracies.append(best_val_acc)
        writer.close()

    # Save category mapping
    with open('category_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(idx_to_category, f, ensure_ascii=False, indent=2)

    # Summarize results
    mean_acc = sum(fold_val_accuracies) / len(fold_val_accuracies)
    logger.info(f"Cross-validation complete. Mean accuracy: {mean_acc:.2f}%")
    logger.info(f"Fold accuracies: {fold_val_accuracies}")

    return fold_val_accuracies, idx_to_category


def ensemble_from_folds(model_class, num_classes, model_paths, device, model_name='efficientnet_b0'):
    """Create an ensemble models from the k-fold trained models"""
    models = []
    for path in model_paths:
        model = model_class(num_classes=num_classes, model_name=model_name).to(device)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        models.append(model)

    return models


def predict_with_ensemble(ensemble_models, image, transform, device):
    """Make a prediction using an ensemble of models"""
    # Prepare image
    if isinstance(image, str) or isinstance(image, Path):
        image = Image.open(image).convert('RGB')

    image_np = np.array(image)
    transformed = transform(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    # Get predictions from all models
    all_probs = []
    with torch.no_grad():
        for model in ensemble_models:
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs)

    # Average probabilities (ensemble method)
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)

    # Get top prediction
    pred_class = torch.argmax(avg_probs, dim=1).item()
    confidence = avg_probs[0][pred_class].item()

    return pred_class, confidence


def main():
    # Configure device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {DEVICE}")

    # Configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 3e-4
    NUM_FOLDS = 5
    MODEL_NAME = 'efficientnet_b0'  # Options: efficientnet_b0, resnet50, mobilenetv3_large_100

    # Load and reorganize data
    with open('dataset/raw_images/labels_reorganized.json', 'r', encoding='utf-8') as f:
        labels_data = json.load(f)

    # Convert to paths and labels
    image_paths = [Path(path) for path in labels_data.keys()]
    labels = list(labels_data.values())

    # Validate images to avoid runtime errors
    valid_image_paths, invalid_image_paths = validate_images(image_paths)
    valid_labels = [labels[image_paths.index(path)] for path in valid_image_paths]

    logger.info(f"Training with {len(valid_image_paths)} valid images across {len(set(valid_labels))} categories")

    # Save invalid images list for reference
    with open('invalid_images.json', 'w', encoding='utf-8') as f:
        json.dump([str(p) for p in invalid_image_paths], f, ensure_ascii=False, indent=2)

    # Train with k-fold cross-validation
    fold_accuracies, category_mapping = train_with_kfold(
        model_class=AdvancedProductClassifier,
        image_paths=valid_image_paths,
        labels=valid_labels,
        num_folds=NUM_FOLDS,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        model_name=MODEL_NAME
    )

    # Find the best fold models
    model_paths = [f'model_fold_{i}_best.pth' for i in range(NUM_FOLDS)]

    # Create ensemble from the trained models
    ensemble_models = ensemble_from_folds(
        model_class=AdvancedProductClassifier,
        num_classes=len(category_mapping),
        model_paths=model_paths,
        device=DEVICE,
        model_name=MODEL_NAME
    )

    # Save final ensemble models
    # Create an average of the weights for simplicity in deployment
    final_model = AdvancedProductClassifier(num_classes=len(category_mapping), model_name=MODEL_NAME).to(DEVICE)

    # Average the models weights (a simple ensemble approach)
    with torch.no_grad():
        for key in final_model.state_dict():
            # Sum of weights divided by number of models
            final_model.state_dict()[key].copy_(
                sum(model.state_dict()[key] for model in ensemble_models) / len(ensemble_models)
            )

    # Save the final models
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'category_mapping': category_mapping,
        'fold_accuracies': fold_accuracies,
        'model_name': MODEL_NAME
    }, 'ensemble_model.pth')

    logger.info("Training completed! Ensemble models saved as 'ensemble_model.pth'")


if __name__ == "__main__":
    main()