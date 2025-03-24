"""
Точка входа для запуска краулера.

Пример использования:
    python -m crawler.main --source wildberries
    python -m crawler.main --source ozon
"""
import os
import json
import time
import argparse
from .logging_setup import logger
from .ozon_crawler import OzonCrawler
from .wildberries_crawler import WildberriesCrawler
from .storage.json_storage import JsonStorage


def parse_arguments():
    """
    Парсит аргументы командной строки.

    Returns:
        argparse.Namespace: Объект с аргументами командной строки.
    """
    parser = argparse.ArgumentParser(description='Краулер Wildberries для сбора данных товаров.')

    parser.add_argument('--max-workers', type=int, default=8,
                        help='Максимальное количество рабочих потоков (по умолчанию: 8)')

    parser.add_argument('--categories-file', type=str, default='categories.json',
                        help='Путь к JSON-файлу со списком категорий (по умолчанию: categories.json)')

    parser.add_argument('--output-dir', type=str, default=None,
                        help='Директория для сохранения результатов (по умолчанию: ~/shared_crawler_output)')

    parser.add_argument('--scale-factor', type=float, default=1.0,
                        help='Коэффициент масштабирования для целевых значений категорий (по умолчанию: 1.0)')

    parser.add_argument('--source', type=str, default="",
                        help='Тип площадки, откуда собирается информация')

    return parser.parse_args()


def calculate_targets(category_targets, scale_factor=1.0, storage=None):
    """
    Рассчитывает целевые значения для категорий с учетом уже собранных товаров.

    Args:
        category_targets (dict): Исходные целевые значения для категорий.
        scale_factor (float): Коэффициент масштабирования (1.0 = 100%).
        storage (JsonStorage): Объект хранилища для получения существующих товаров.

    Returns:
        dict: Скорректированные целевые значения для категорий.
    """
    # Если не задан объект хранилища, создаем новый
    if storage is None:
        storage = JsonStorage()

    # Получаем количество уже собранных товаров по категориям
    existing_category_counts = storage.get_category_counts()

    # Рассчитываем скорректированные целевые значения
    adjusted_targets = {}
    for category, target in category_targets.items():
        # Масштабируем целевое значение
        scaled_target = int(target * scale_factor)

        # Вычитаем уже собранные товары
        existing_count = existing_category_counts.get(category, 0)
        adjusted_target = max(0, scaled_target - existing_count)

        adjusted_targets[category] = adjusted_target

    return adjusted_targets


def load_categories_from_file(file_path):
    """
    Загружает целевые значения категорий из JSON-файла.

    Args:
        file_path (str): Путь к JSON-файлу с категориями.

    Returns:
        dict: Словарь с целевыми значениями для категорий или пустой словарь в случае ошибки.
    """
    if not os.path.exists(file_path):
        logger.error(f"Файл категорий не найден: {file_path}")
        return {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            categories = json.load(f)

        logger.info(f"Загружены категории из файла {file_path}: {len(categories)} категорий")
        return categories
    except json.JSONDecodeError:
        logger.error(f"Ошибка декодирования JSON в файле: {file_path}")
        return {}
    except Exception as e:
        logger.error(f"Ошибка при чтении файла категорий {file_path}: {e}")
        return {}


def get_default_categories():
    """
    Возвращает словарь категорий по умолчанию, если файл категорий не найден.

    Returns:
        dict: Словарь с целевыми значениями для категорий по умолчанию.
    """
    default_categories = {
        "Презервативы": 50,
        "Вибраторы": 50,
        "Фаллоимитаторы": 50,
        "Анальные пробки": 50,
        "Мастурбаторы мужские": 50
    }

    logger.info("Используются категории по умолчанию")
    return default_categories


def main():
    """
    Основная функция для запуска краулера.
    """
    # Парсим аргументы командной строки
    args = parse_arguments()

    # Если указана выходная директория, создаем хранилище с этой директорией
    storage = None
    if args.output_dir:
        storage = JsonStorage(args.output_dir)

    # Загружаем категории из файла или используем значения по умолчанию
    category_targets = load_categories_from_file(args.categories_file)
    if not category_targets:
        category_targets = get_default_categories()

    # Рассчитываем целевые значения с учетом масштабирования и существующих товаров
    adjusted_targets = calculate_targets(category_targets, args.scale_factor, storage)

    # Выводим информацию о целевых значениях
    logger.info(f"Всего категорий: {len(adjusted_targets)}")
    logger.info(f"Общее целевое количество товаров: {sum(adjusted_targets.values())}")

    # Фильтруем категории с нулевыми целевыми значениями
    adjusted_targets = {category: target for category, target in adjusted_targets.items() if target > 0}

    if not adjusted_targets:
        logger.warning("Нет категорий для сбора (все целевые значения равны 0)")
        return

        # Выбираем и создаем краулер в зависимости от указанного источника
    source = args.source.lower()
    logger.info(f"Запуск краулера для источника: {source}")

    if source == 'ozon':
        # Создаем краулер Ozon
        crawler = OzonCrawler(adjusted_targets, max_workers=args.max_workers, output_dir=args.output_dir)
    else:
        # По умолчанию используем Wildberries
        crawler = WildberriesCrawler(adjusted_targets, max_workers=args.max_workers, output_dir=args.output_dir)

    try:
        start_time = time.time()
        crawler.run()
        end_time = time.time()

        # Выводим статистику
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        logger.info(f"Краулинг завершен за {int(hours)}:{int(minutes):02}:{int(seconds):02}")
        logger.info(f"Обработано категорий: {len(crawler.processed_categories)}")
        logger.info(f"Собрано ссылок: {crawler.urls_collected}")
        logger.info(f"Обработано товаров: {crawler.products_processed}")

    except KeyboardInterrupt:
        logger.info("Краулинг прерван пользователем (Ctrl+C)")
    except Exception as e:
        logger.error(f"Произошла ошибка во время выполнения краулера: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()