import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt


class DatasetReorganizer:
    def __init__(self, labels_path):
        self.labels_path = labels_path
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)

        self.category_mapping = {
            'Вибраторы': ['Вибраторы', 'Вибратор', 'Вибропули', 'Виброяйца', 'Вибротрусики'],
            'Фаллоимитаторы': ['Фаллоимитаторы', 'Фаллоимитатор', 'Фаллопротезы'],
            'Анальные игрушки': ['Анальные пробки', 'Анальная пробка', 'Анальные шарики',
                                 'Анальные бусы', 'Массажеры простаты', 'Анальные груши'],
            'БДСМ': ['БДСМ аксессуар', 'Комплекты БДСМ', 'Наручники эротик', 'Кляпы эротик',
                     'Ошейники эротик', 'Стэки эротик', 'Плетки эротик', 'Шлепалки эротик',
                     'Оковы эротик', 'Пояса верности', 'Фиксаторы эротик', 'Зажимы для сосков'],
            'Страпоны': ['Страпоны', 'Страпон', 'Насадки на страпон', 'Трусы для страпона'],
            'Мастурбаторы': ['Мастурбаторы мужские', 'Мастурбатор'],
            'Эротическое белье': ['Пеньюары эротик', 'Трусы эротик', 'Комплекты эротик',
                                  'Боди эротик', 'Корсеты эротик', 'Комбинезоны эротик'],
            'Эротическая одежда': ['Платья эротик', 'Ролевые костюмы эротик', 'Чулки эротик',
                                   'Колготки эротик'],
            'Аксессуары': ['Насадки на член', 'Эрекционные кольца', 'Вакуумные помпы эротик'],
            'Интимная косметика': ['Лубриканты', 'Массажные средства эротик',
                                   'Возбуждающие средства', 'Презервативы']
        }

    def remap_categories(self):
        new_labels = {}
        skipped = 0

        reverse_mapping = {}
        for new_cat, old_cats in self.category_mapping.items():
            for old_cat in old_cats:
                reverse_mapping[old_cat] = new_cat

        for image_path, old_category in self.labels.items():
            if old_category in reverse_mapping:
                new_labels[image_path] = reverse_mapping[old_category]
            else:
                skipped += 1
                continue

        print(f"Пропущено изображений: {skipped}")
        return new_labels

    def save_new_labels(self, new_labels, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_labels, f, ensure_ascii=False, indent=2)

    def analyze_distribution(self, labels):
        counts = Counter(labels.values())
        print("\nНовое распределение по категориям:")
        for category, count in counts.most_common():
            print(f"{category}: {count} изображений")

        plt.figure(figsize=(12, 6))
        categories = [cat for cat, _ in counts.most_common()]
        counts_values = [count for _, count in counts.most_common()]

        plt.bar(categories, counts_values)
        plt.xticks(rotation=45, ha='right')
        plt.title('Новое распределение изображений по категориям')
        plt.ylabel('Количество изображений')
        plt.tight_layout()
        plt.savefig('new_distribution.png')
        plt.close()


def main():
    reorganizer = DatasetReorganizer("dataset/raw_images/labels.json")
    new_labels = reorganizer.remap_categories()
    reorganizer.save_new_labels(new_labels, "dataset/raw_images/labels_reorganized.json")
    reorganizer.analyze_distribution(new_labels)


if __name__ == "__main__":
    main()