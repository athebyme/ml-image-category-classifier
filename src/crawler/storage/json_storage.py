"""
Модуль для работы с хранилищем данных в формате JSON.
"""
import os
import json
from ..logging_setup import logger
from ..config import OUTPUT_DIR


class JsonStorage:
    """
    Класс для управления хранилищем данных в формате JSON.

    Обеспечивает сохранение, загрузку и управление JSON-данными.
    """

    def __init__(self, output_dir=None):
        """
        Инициализирует хранилище JSON.

        Args:
            output_dir (str, optional): Директория для хранения файлов JSON.
        """
        self.output_dir = output_dir or OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.existing_articles = self.load_existing_articles()

    def save_product(self, product_data):
        """
        Сохраняет данные о товаре в JSON-файл.

        Args:
            product_data (dict): Данные о товаре.

        Returns:
            bool: True, если данные сохранены успешно, иначе False.
        """
        if product_data and product_data.get('article'):
            article = product_data['article']
            filename = f"product_{article}.json"
            filepath = os.path.join(self.output_dir, filename)
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(product_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Сохранён товар {article} в {filepath}")
                self.existing_articles.add(article)
                return True
            except Exception as e:
                logger.error(f"Ошибка сохранения данных товара {article} в {filepath}: {e}")
                return False
        else:
            logger.warning("Данные товара или номер статьи отсутствуют. Сохранение пропущено.")
            return False

    def load_existing_articles(self):
        """
        Загружает артикулы из уже собранных JSON-файлов.

        Returns:
            set: Набор существующих артикулов.
        """
        existing_articles = set()
        if os.path.exists(self.output_dir) and os.path.isdir(self.output_dir):
            for filename in os.listdir(self.output_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.output_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            article = data.get("article")
                            if article:
                                existing_articles.add(article)
                    except (json.JSONDecodeError, FileNotFoundError) as e:
                        logger.warning(f"Ошибка при чтении файла {filename}: {e}")
        logger.info(f"Загружено {len(existing_articles)} существующих артикулов.")
        return existing_articles

    def get_existing_articles(self):
        """
        Возвращает набор существующих артикулов.

        Returns:
            set: Набор существующих артикулов.
        """
        return self.existing_articles

    def save_state(self, state_data):
        """
        Сохраняет состояние краулера.

        Args:
            state_data (dict): Данные о состоянии.

        Returns:
            bool: True, если данные сохранены успешно, иначе False.
        """
        try:
            state_file = os.path.join(self.output_dir, "crawler_state.json")
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)

            logger.info("Состояние краулера сохранено в crawler_state.json")
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении состояния: {e}")
            return False

    def load_state(self):
        """
        Загружает состояние краулера.

        Returns:
            dict: Данные о состоянии или None, если не удалось загрузить.
        """
        state_file = os.path.join(self.output_dir, "crawler_state.json")
        if not os.path.exists(state_file):
            return None

        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state_data = json.load(f)

            logger.info("Загружено состояние краулера из crawler_state.json")
            return state_data
        except Exception as e:
            logger.error(f"Ошибка при загрузке состояния: {e}")
            return None

    def get_category_counts(self):
        """
        Подсчитывает количество товаров по категориям.

        Returns:
            dict: Словарь с количеством товаров по категориям.
        """
        category_counts = {}

        if os.path.exists(self.output_dir) and os.path.isdir(self.output_dir):
            for filename in os.listdir(self.output_dir):
                if filename.endswith(".json") and filename != "crawler_state.json":
                    filepath = os.path.join(self.output_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            category = data.get("product-category-description")
                            if category:
                                category_counts[category] = category_counts.get(category, 0) + 1
                    except json.JSONDecodeError:
                        logger.error(f"Ошибка декодирования JSON в файле: {filename}")
                    except Exception as e:
                        logger.error(f"Ошибка при чтении файла {filename}: {e}")

        return category_counts