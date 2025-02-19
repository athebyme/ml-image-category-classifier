import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt


def analyze_dataset(labels_path):
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    category_counts = Counter(labels.values())

    print("\nДатасет содержит:")
    print(f"Всего изображений: {len(labels)}")
    print("\nРаспределение по категориям:")
    for category, count in category_counts.most_common():
        print(f"{category}: {count} изображений")

    missing_files = []
    for image_path in labels.keys():
        if not Path(image_path).exists():
            missing_files.append(image_path)

    if missing_files:
        print("\nВнимание! Не найдены следующие файлы:")
        for path in missing_files[:5]:
            print(path)
        if len(missing_files) > 5:
            print(f"...и еще {len(missing_files) - 5} файлов")

    plt.figure(figsize=(12, 6))
    categories = [cat for cat, _ in category_counts.most_common()]
    counts = [count for _, count in category_counts.most_common()]

    plt.bar(categories, counts)
    plt.xticks(rotation=45, ha='right')
    plt.title('Распределение изображений по категориям')
    plt.ylabel('Количество изображений')
    plt.tight_layout()
    plt.savefig('dataset_distribution.png')
    plt.close()


if __name__ == "__main__":
    labels_path = "dataset/raw_images/labels.json"
    analyze_dataset(labels_path)