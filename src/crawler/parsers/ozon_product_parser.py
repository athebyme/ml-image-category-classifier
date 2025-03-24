"""
Модуль для парсинга страниц товаров Ozon.
"""
import re
import time
import json
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from ..logging_setup import logger


class OzonProductParser:
    """
    Класс для парсинга страниц товаров Ozon.

    Извлекает информацию о товаре, включая название, бренд, описание,
    характеристики и изображения.
    """

    def __init__(self, browser_manager, captcha_solver=None):
        """
        Инициализирует парсер товаров Ozon.

        Args:
            browser_manager (BrowserManager): Менеджер браузера.
            captcha_solver (CaptchaSolver, optional): Решатель CAPTCHA.
        """
        self.browser_manager = browser_manager
        self.captcha_solver = captcha_solver

    def process_product_page(self, driver, url, category):
        """
        Обрабатывает страницу товара Ozon и извлекает данные.

        Args:
            driver (WebDriver): Экземпляр веб-драйвера.
            url (str): URL страницы товара.
            category (str): Категория товара.

        Returns:
            dict: Данные о товаре или None в случае ошибки.
        """
        try:
            logger.info(f"Обработка карточки товара Ozon: {url}")

            # Загружаем страницу
            driver.get(url)
            time.sleep(2)  # Даем время на начальную загрузку

            # Проверяем на наличие CAPTCHA
            if self.captcha_solver and self.captcha_solver.detect_captcha(driver):
                logger.warning(f"Обнаружена CAPTCHA на странице товара Ozon")
                if not self.captcha_solver.handle_captcha(driver):
                    logger.error(f"Не удалось решить CAPTCHA на странице товара Ozon")
                    return None

            # Ждем загрузку основной информации
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "h1.te"))  # Селектор заголовка Ozon
                )
            except TimeoutException:
                logger.error(f"Таймаут при ожидании загрузки страницы товара Ozon: {url}")
                return None

            # Имитируем действия пользователя
            self.browser_manager.simulate_human_behavior(driver)

            # Прокручиваем страницу для загрузки всех элементов
            self.browser_manager.smooth_scroll(driver)

            # Используем BeautifulSoup для парсинга
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Извлекаем артикул из URL
            article = self.extract_article_from_url(url)

            # Название товара
            try:
                name_element = soup.select_one("h1.te")
                name = name_element.text.strip() if name_element else ""
            except Exception as e:
                logger.error(f"Ошибка при извлечении названия товара Ozon: {e}")
                name = ""

            # Бренд
            try:
                brand_element = soup.select_one("a[data-test-id='brand-link']")
                if not brand_element:
                    brand_element = soup.select_one("a[data-widget='webBrand']")
                brand = brand_element.text.strip() if brand_element else ""
            except Exception as e:
                logger.error(f"Ошибка при извлечении бренда товара Ozon: {e}")
                brand = ""

            # Цена
            try:
                price_element = soup.select_one("span[data-test-id='price']")
                if not price_element:
                    price_element = soup.select_one("span.tsBody500Medium")
                price = price_element.text.strip() if price_element else ""
            except Exception as e:
                logger.error(f"Ошибка при извлечении цены товара Ozon: {e}")
                price = ""

            # Описание товара
            try:
                description_element = soup.select_one("div[data-widget='webDescription']")
                description = description_element.text.strip() if description_element else ""
            except Exception as e:
                logger.error(f"Ошибка при извлечении описания товара Ozon: {e}")
                description = ""

            # Характеристики товара
            characteristics = {}
            try:
                # Прокручиваем к характеристикам
                try:
                    char_section = driver.find_element(By.CSS_SELECTOR, "div[data-widget='webCharacteristics']")
                    driver.execute_script("arguments[0].scrollIntoView(true);", char_section)
                    time.sleep(1)
                except:
                    pass

                # Парсим характеристики
                char_rows = soup.select("div[data-widget='webCharacteristics'] dl")
                for row in char_rows:
                    key_element = row.select_one("dt")
                    val_element = row.select_one("dd")
                    if key_element and val_element:
                        key = key_element.text.strip()
                        value = val_element.text.strip()
                        characteristics[key] = value
            except Exception as e:
                logger.error(f"Ошибка при извлечении характеристик товара Ozon: {e}")

            # Хлебные крошки
            breadcrumbs = []
            try:
                breadcrumb_elements = soup.select("div[data-widget='breadCrumbs'] a")
                for bc in breadcrumb_elements:
                    breadcrumbs.append(bc.text.strip())
            except Exception as e:
                logger.error(f"Ошибка при извлечении хлебных крошек товара Ozon: {e}")

            # Рейтинг
            rating = ""
            try:
                rating_element = soup.select_one("div[data-widget='webReviewProductScore'] span.tsBodyControl400Medium")
                rating = rating_element.text.strip() if rating_element else ""
            except Exception as e:
                logger.error(f"Ошибка при извлечении рейтинга товара Ozon: {e}")

            # Изображения товара
            images = []
            try:
                # Сначала ищем в JSON-данных
                image_data = self.extract_images_from_json(driver.page_source)
                if image_data:
                    images = image_data

                # Если не нашли в JSON, пробуем через DOM
                if not images:
                    image_elements = soup.select("img.ij")
                    for img in image_elements:
                        src = img.get("src")
                        if src and "ozon" in src:
                            # Преобразуем URL к высокому разрешению
                            if "wc50" in src:
                                src = src.replace("wc50", "wc1000")
                            elif "wc250" in src:
                                src = src.replace("wc250", "wc1000")
                            images.append(src)
            except Exception as e:
                logger.error(f"Ошибка при извлечении изображений товара Ozon: {e}")

            # Если не нашли изображения, попробуем через селениум
            if not images:
                try:
                    image_elements = driver.find_elements(By.CSS_SELECTOR, "img.ij")
                    for img in image_elements:
                        src = img.get_attribute("src")
                        if src and "ozon" in src:
                            # Преобразуем URL к высокому разрешению
                            if "wc50" in src:
                                src = src.replace("wc50", "wc1000")
                            elif "wc250" in src:
                                src = src.replace("wc250", "wc1000")
                            images.append(src)
                except Exception as e:
                    logger.error(f"Ошибка при извлечении изображений через Selenium: {e}")

            # Проверяем, нашли ли мы основные данные
            if not name:
                logger.error(f"Не удалось извлечь название товара Ozon: {url}")
                return None

            # Создаем структуру данных
            product_data = {
                "article": article,
                "name": name,
                "brand": brand,
                "price": price,
                "rating": rating,
                "breadcrumbs": breadcrumbs,
                "description": description,
                "characteristics": characteristics,
                "images": images,
                "product-category-description": category,
                "url": url,
                "source": "ozon"  # Добавляем маркер источника
            }

            logger.success(f"Успешно обработан товар Ozon: {article} - {name}")
            return product_data

        except Exception as e:
            logger.error(f"Ошибка при обработке страницы товара Ozon {url}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def extract_article_from_url(self, url):
        """
        Извлекает артикул товара из URL Ozon.

        Args:
            url (str): URL страницы товара.

        Returns:
            str: Артикул товара или None, если не удалось извлечь.
        """
        # Формат URL Ozon: https://www.ozon.ru/product/название-товара-12345678/
        try:
            match = re.search(r'/product/[^/]+-(\d+)/?', url)
            if match:
                return match.group(1)

            # Альтернативный формат
            match = re.search(r'/context/detail/id/(\d+)/?', url)
            if match:
                return match.group(1)

            # Если не удалось найти по шаблону, берем последнее число из URL
            numbers = re.findall(r'(\d+)', url)
            if numbers:
                return numbers[-1]

            return None
        except Exception as e:
            logger.error(f"Ошибка при извлечении артикула из URL Ozon: {e}")
            return None

    def extract_images_from_json(self, page_source):
        """
        Извлекает URL изображений из JSON-данных на странице.

        Args:
            page_source (str): Исходный код страницы.

        Returns:
            list: Список URL изображений.
        """
        try:
            # Ищем JSON-данные с изображениями в исходном коде страницы
            json_pattern = r'<script type="application/json" id="state-pdp">(.+?)</script>'
            json_match = re.search(json_pattern, page_source, re.DOTALL)

            if json_match:
                json_data = json_match.group(1)
                data = json.loads(json_data)

                images = []

                # Пытаемся найти изображения в различных структурах JSON
                for key in data:
                    if "gallery" in key.lower():
                        gallery_data = data[key]
                        if isinstance(gallery_data, dict) and "images" in gallery_data:
                            for img in gallery_data["images"]:
                                if isinstance(img, dict) and "src" in img:
                                    img_url = img["src"]
                                    # Преобразуем к высокому разрешению
                                    if "wc50" in img_url:
                                        img_url = img_url.replace("wc50", "wc1000")
                                    elif "wc250" in img_url:
                                        img_url = img_url.replace("wc250", "wc1000")
                                    images.append(img_url)

                # Если не нашли через gallery, поищем другие возможные структуры
                if not images:
                    # Рекурсивный поиск URL изображений в JSON
                    images = self._find_image_urls_in_json(data)

                return images

            return []
        except Exception as e:
            logger.error(f"Ошибка при извлечении изображений из JSON: {e}")
            return []

    def _find_image_urls_in_json(self, data, max_depth=5, current_depth=0):
        """
        Рекурсивно ищет URL изображений в JSON-структуре.

        Args:
            data: JSON-данные для поиска.
            max_depth (int): Максимальная глубина рекурсии.
            current_depth (int): Текущая глубина рекурсии.

        Returns:
            list: Список URL изображений.
        """
        if current_depth >= max_depth:
            return []

        image_urls = []

        if isinstance(data, dict):
            for key, value in data.items():
                # Проверяем ключи, которые могут содержать URL изображений
                if isinstance(key, str) and ("image" in key.lower() or "photo" in key.lower() or "src" == key.lower()):
                    if isinstance(value, str) and (
                            "ozon" in value and ("jpg" in value or "png" in value or "jpeg" in value)):
                        image_urls.append(value)

                # Рекурсивно ищем в значениях
                if isinstance(value, (dict, list)):
                    image_urls.extend(self._find_image_urls_in_json(value, max_depth, current_depth + 1))

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    image_urls.extend(self._find_image_urls_in_json(item, max_depth, current_depth + 1))

        return image_urls