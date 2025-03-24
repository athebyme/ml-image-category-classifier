"""
Модуль для парсинга страниц товаров Wildberries.
"""
import re
import time
import base64
import requests
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from ..logging_setup import logger


class ProductParser:
    """
    Класс для парсинга страниц товаров Wildberries.

    Извлекает информацию о товаре, включая название, бренд, описание,
    характеристики и изображения.
    """

    def __init__(self, browser_manager, captcha_solver=None):
        """
        Инициализирует парсер товаров.

        Args:
            browser_manager (BrowserManager): Менеджер браузера.
            captcha_solver (CaptchaSolver, optional): Решатель CAPTCHA.
        """
        self.browser_manager = browser_manager
        self.captcha_solver = captcha_solver

    def process_product_page(self, driver, url, category):
        """
        Обрабатывает страницу товара и извлекает данные.

        Args:
            driver (WebDriver): Экземпляр веб-драйвера.
            url (str): URL страницы товара.
            category (str): Категория товара.

        Returns:
            dict: Данные о товаре или None в случае ошибки.
        """
        try:
            logger.info(f"Обработка карточки товара: {url}")

            # Проверяем, загрузилась ли страница
            try:
                # Ждем загрузку основной информации о товаре
                WebDriverWait(driver, 20).until(
                    EC.visibility_of_element_located((By.CLASS_NAME, "product-page__title"))
                )

                try:
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_all_elements_located(
                            (By.CSS_SELECTOR, "ul.breadcrumbs__list li.breadcrumbs__item"))
                    )
                except TimeoutException:
                    logger.warning(f"Таймаут при ожидании хлебных крошек на {url}")
            except TimeoutException:
                logger.error(f"Таймаут при ожидании заголовка товара на {url}")
                return None

            # Парсим основную информацию
            main_soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Название товара
            name_element = main_soup.select_one("h1.product-page__title")
            name = name_element.text.strip() if name_element else ""

            # Бренд
            brand_element = main_soup.select_one("a.product-page__header-brand")
            brand = brand_element.text.strip() if brand_element else ""

            # Хлебные крошки (навигация)
            breadcrumb_elements = main_soup.select("ul.breadcrumbs__list li.breadcrumbs__item span[itemprop='name']")

            # Если первая попытка не удалась, пробуем альтернативные селекторы
            if not breadcrumb_elements:
                breadcrumb_elements = main_soup.select("ul.breadcrumbs__list li.breadcrumbs__item")
                breadcrumbs = [crumb.text.strip() for crumb in breadcrumb_elements if crumb.text.strip()]
                logger.info(f"Использован альтернативный селектор хлебных крошек, найдено {len(breadcrumbs)} элементов")
            else:
                breadcrumbs = [crumb.text.strip() for crumb in breadcrumb_elements]
                logger.info(f"Использован оригинальный селектор хлебных крошек, найдено {len(breadcrumbs)} элементов")

            # Начальные изображения (низкого качества, если есть)
            initial_images = []
            for img in main_soup.select("div.slide__content img.photo-zoom__preview"):
                src = img.get('src')
                if src:
                    if src.startswith('//'):
                        src = 'https:' + src
                    initial_images.append(src)

            # Описание и характеристики из всплывающего окна
            description = ""
            characteristics = {}

            try:
                # Ищем и нажимаем на кнопку "Подробнее о товаре"
                details_btn = WebDriverWait(driver, 20).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button.product-page__btn-detail"))
                )
                driver.execute_script("arguments[0].scrollIntoView(true);", details_btn)
                time.sleep(0.5)

                try:
                    details_btn.click()
                except Exception as e:
                    driver.execute_script("arguments[0].click();", details_btn)

                # Ждем загрузки всплывающего окна
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "table.product-params__table"))
                )
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "section.product-details__description p.option__text"))
                )

                # Даем время на полную загрузку всплывающего окна
                time.sleep(1)

                # Парсим содержимое всплывающего окна
                popup_soup = BeautifulSoup(driver.page_source, 'html.parser')

                # Характеристики товара
                for row in popup_soup.select("table.product-params__table tr"):
                    key_elem = row.select_one("th.product-params__cell")
                    val_elem = row.select_one("td.product-params__cell")
                    if key_elem and val_elem:
                        characteristics[key_elem.text.strip()] = val_elem.text.strip()

                # Описание товара - ищем по заголовку "Описание"
                description_section = popup_soup.find("section", class_="product-details__description")
                if description_section:
                    description_p = description_section.find("p", class_="option__text")
                    if description_p:
                        description = description_p.text.strip()
                    else:
                        logger.debug("Параграф с описанием не найден")
                else:
                    logger.debug("Секция с описанием не найдена")

                # Закрываем всплывающее окно
                try:
                    close_btn = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "a.j-close.popup__close.close"))
                    )
                    close_btn.click()
                except Exception as e:
                    logger.debug(f"Не удалось закрыть всплывающее окно: {e}")

            except Exception as e:
                logger.error(f"Ошибка при работе с всплывающим окном характеристик: {e}")

            # Получаем изображения высокого качества
            images = self.parse_wb_slider_images_high_quality(driver)

            # Если не нашли изображения высокого качества, используем начальные
            if not images:
                images = initial_images

            # Извлекаем артикул из URL
            article = self.extract_article_from_url(url)
            logger.info(f"Успешно обработан товар: {article} - {name}")

            return {
                "article": article,
                "name": name,
                "brand": brand,
                "breadcrumbs": breadcrumbs,
                "description": description,
                "characteristics": characteristics,
                "images": images,
                "product-category-description": category,
                "url": url
            }
        except Exception as e:
            logger.error(f"Ошибка обработки карточки товара {url}: {e}")
            return None

    def extract_article_from_url(self, url):
        """
        Извлекает артикул товара из URL.

        Args:
            url (str): URL страницы товара.

        Returns:
            str: Артикул товара или None, если не удалось извлечь.
        """
        match = re.search(r'/catalog/(\d+)/detail\.aspx', url)
        if match:
            return match.group(1)
        return None

    def download_image(self, url):
        """
        Загружает изображение по URL.

        Args:
            url (str): URL изображения.

        Returns:
            str: Base64-кодированное содержимое изображения или None в случае ошибки.
        """
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                logger.debug(f"Изображение успешно загружено: {url}")
                return base64.b64encode(response.content).decode('utf-8')
        except requests.exceptions.Timeout:
            logger.error(f"Таймаут при загрузке изображения {url}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке изображения {url}: {e}")
        return None

    def save_data_url_to_file(self, data_url, product_id):
        """
        Сохраняет data URL как файл и возвращает путь к файлу.

        Args:
            data_url (str): Data URL изображения.
            product_id (str): Идентификатор товара.

        Returns:
            str: Путь к сохраненному файлу или None в случае ошибки.
        """
        try:
            import base64
            import os
            from datetime import datetime

            # Создаем директорию, если она не существует
            img_dir = os.path.join("images", str(product_id))
            os.makedirs(img_dir, exist_ok=True)

            # Извлекаем base64-кодированные данные
            header, encoded = data_url.split(",", 1)
            data = base64.b64decode(encoded)

            # Генерируем уникальное имя файла
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            file_path = os.path.join(img_dir, f"image_{timestamp}.png")

            # Записываем в файл
            with open(file_path, "wb") as f:
                f.write(data)

            return file_path
        except Exception as e:
            logger.error(f"Ошибка при сохранении data URL: {e}")
            return None

    def parse_wb_slider_images(self, driver):
        """
        Парсит все изображения из слайдера Wildberries.

        Args:
            driver (WebDriver): Экземпляр веб-драйвера.

        Returns:
            list: Список URL изображений.
        """
        image_urls = set()
        max_attempts = 20
        attempts = 0

        while attempts < max_attempts:
            try:
                # Ждем загрузки слайдера
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "ul.swiper-wrapper"))
                )

                # Получаем все слайды на текущей странице
                slides = driver.find_elements(By.CSS_SELECTOR, "li.swiper-slide.j-product-photo")

                images_before = len(image_urls)

                # Обрабатываем каждый слайд
                for slide in slides:
                    try:
                        # Проверяем, что слайд видим
                        if not slide.is_displayed():
                            continue

                        # Находим изображение внутри слайда
                        img = slide.find_element(By.CSS_SELECTOR, "div.slide__content img")
                        src = img.get_attribute('data-src-pb')

                        if src:
                            if src.startswith('//'):
                                src = 'https:' + src
                            image_urls.add(src)

                    except Exception as e:
                        logger.debug(f"Ошибка при обработке слайда: {e}")
                        continue

                # Если не появилось новых изображений, пробуем пролистнуть
                if len(image_urls) == images_before:
                    try:
                        next_button = WebDriverWait(driver, 2).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.swiper-button-next"))
                        )
                        # Проверяем, активна ли кнопка
                        if 'swiper-button-disabled' in next_button.get_attribute('class'):
                            break

                        driver.execute_script("arguments[0].click();", next_button)
                        time.sleep(0.5)  # Ждем анимацию слайдера

                        # Если после клика количество изображений не изменилось и прошло 2 попытки,
                        # значит, мы достигли конца слайдера
                        if len(image_urls) == images_before and attempts > 1:
                            break

                    except TimeoutException:
                        logger.debug("Кнопка Next не найдена или недоступна")
                        break

                attempts += 1

            except Exception as e:
                logger.error(f"Критическая ошибка при работе со слайдером: {e}")
                break

        return list(image_urls)

    def parse_wb_slider_images_high_quality(self, driver):
        """
        Улучшенный метод для парсинга изображений высокого качества с Wildberries.

        Args:
            driver (WebDriver): Экземпляр веб-драйвера.

        Returns:
            list: Список URL изображений высокого качества.
        """
        image_urls = set()
        max_attempts = 15
        attempts = 0

        try:
            # Сначала определяем, с каким типом галереи изображений мы имеем дело
            # Wildberries имеет несколько макетов галереи в зависимости от товара

            # Ждем загрузки галереи/слайдера
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((
                        By.CSS_SELECTOR,
                        "ul.swiper-wrapper, div.slider-block, div.sw-slider-product, div.zoom-image-container"
                    ))
                )
                time.sleep(1)  # Дополнительное время, чтобы изображения полностью загрузились
            except TimeoutException:
                logger.warning("Не найдена галерея изображений на странице товара")
                return []

            # Сначала пытаемся найти большие изображения напрямую в исходном коде страницы
            # Это самый надежный и высококачественный метод
            page_source = driver.page_source

            # Шаблоны для URL изображений высокого качества в исходном коде
            patterns = [
                r'(https:\/\/images\.wbstatic\.net\/big\/new\/\d+\/\d+[-\w]+\.jpg)',
                r'(https:\/\/images\.wbstatic\.net\/large\/new\/\d+\/\d+[-\w]+\.jpg)',
                r'(https:\/\/images\.wbstatic\.net\/c516x688\/new\/\d+\/\d+[-\w]+\.jpg)',
                r'(https:\/\/[\w-]+\.wbstatic\.net\/big\/new\/\d+\/\d+.*?\.jpg)'
            ]

            for pattern in patterns:
                big_images = re.findall(pattern, page_source)
                for img_url in big_images:
                    # Убеждаемся, что это полный URL
                    if img_url.startswith('//'):
                        img_url = 'https:' + img_url
                    image_urls.add(img_url)

            # Если мы нашли изображения через regex, возвращаем их
            if image_urls:
                logger.info(f"Найдено {len(image_urls)} изображений высокого качества через анализ HTML")
                return list(image_urls)

            # Если метод regex не сработал, пробуем извлечение через DOM
            logger.info("Попытка извлечения изображений через DOM элементы")

            # Получаем все видимые слайды
            current_strategy = "standard_gallery"
            slides = driver.find_elements(By.CSS_SELECTOR,
                                          "li.swiper-slide.j-product-photo:not([style*='display: none'])")

            if not slides:
                slides = driver.find_elements(By.CSS_SELECTOR, "div.sw-slider-product__item")
                current_strategy = "new_gallery"

            if not slides:
                slides = driver.find_elements(By.CSS_SELECTOR, "div.slider-block__item")
                current_strategy = "alternate_gallery"

            if not slides:
                # В крайнем случае - ищем любое изображение, которое может быть изображением товара
                logger.warning("Не найдены слайды галереи, пробуем найти любые изображения товара")
                current_strategy = "fallback"

                # Пытаемся найти любые изображения товара
                img_elements = driver.find_elements(By.CSS_SELECTOR, "img[src*='/catalog/']")

                for img in img_elements:
                    src = img.get_attribute("src")
                    if src and not src.startswith("data:"):
                        # Преобразуем в URL высокого разрешения
                        high_res_url = src.replace("/tm/", "/big/").replace("/c246x328/", "/big/")
                        image_urls.add(high_res_url)

                # Если мы нашли изображения таким образом, возвращаем их
                if image_urls:
                    return list(image_urls)

            # Обрабатываем слайды в зависимости от обнаруженного типа галереи
            logger.info(f"Используем стратегию извлечения изображений: {current_strategy}")

            # Функция для извлечения URL изображения и преобразования в высокое разрешение
            def extract_and_convert_image_url(element):
                try:
                    img_url = None

                    # Проверяем различные атрибуты, которые могут содержать URL
                    for attr in ["src", "data-src", "data-bx-src", "data-original", "data-src-pb"]:
                        img_url = element.get_attribute(attr)
                        if img_url and not img_url.startswith("data:"):
                            break

                    if not img_url or img_url.startswith("data:"):
                        return None

                    # Нормализуем URL
                    if img_url.startswith('//'):
                        img_url = 'https:' + img_url

                    # Преобразуем в высокое разрешение
                    conversions = [
                        ("/tm/", "/big/"),
                        ("/c246x328/", "/big/"),
                        ("/c252x336/", "/big/"),
                        ("/c516x688/", "/big/"),
                        ("/middle/", "/big/")
                    ]

                    for old, new in conversions:
                        img_url = img_url.replace(old, new)

                    return img_url
                except Exception as e:
                    logger.error(f"Ошибка при извлечении URL изображения: {e}")
                    return None

            # Обрабатываем слайды в зависимости от стратегии
            if current_strategy == "standard_gallery" or current_strategy == "new_gallery":
                # Для каждого слайда в галерее
                for slide in slides:
                    try:
                        # Находим элемент изображения
                        img_element = None
                        try:
                            img_element = slide.find_element(By.CSS_SELECTOR, "img")
                        except:
                            try:
                                img_element = slide.find_element(By.CSS_SELECTOR, "div.slide__content img")
                            except:
                                try:
                                    img_element = slide.find_element(By.CSS_SELECTOR, "div.sw-slider-product__img img")
                                except:
                                    logger.debug(f"Не найдено изображение в слайде")
                                    continue

                        if img_element:
                            img_url = extract_and_convert_image_url(img_element)
                            if img_url:
                                image_urls.add(img_url)
                    except Exception as e:
                        logger.debug(f"Ошибка при обработке слайда: {e}")
                        continue

                    # Если мы еще не нашли изображения, пробуем кликнуть на слайд, чтобы открыть модальное окно
                    if not image_urls and attempts < 5:
                        try:
                            from selenium.webdriver.common.action_chains import ActionChains
                            from selenium.webdriver.common.keys import Keys

                            driver.execute_script("arguments[0].click();", slide)
                            time.sleep(0.5)

                            # Ищем изображения в модальном окне
                            modal_images = driver.find_elements(By.CSS_SELECTOR, "img.photo-zoom__preview")
                            for img in modal_images:
                                img_url = extract_and_convert_image_url(img)
                                if img_url:
                                    image_urls.add(img_url)

                            # Закрываем модальное окно
                            actions = ActionChains(driver)
                            actions.send_keys(Keys.ESCAPE).perform()
                            time.sleep(0.3)
                        except Exception as e:
                            logger.debug(f"Ошибка при попытке кликнуть на слайд: {e}")

                    attempts += 1

                # Если у нас все еще нет изображений, пробуем найти кнопку "next" и перейти к следующим слайдам
                if len(image_urls) < 3 and attempts < max_attempts:  # Мы хотим хотя бы 3 изображения, если возможно
                    try:
                        next_button = WebDriverWait(driver, 2).until(
                            EC.element_to_be_clickable(
                                (By.CSS_SELECTOR, "button.swiper-button-next, div.slider-control--next"))
                        )

                        # Проверяем, отключена ли кнопка
                        if 'swiper-button-disabled' in next_button.get_attribute('class'):
                            logger.debug("Кнопка Next заблокирована, все слайды просмотрены.")
                        else:
                            # Кликаем и ждем новых слайдов
                            driver.execute_script("arguments[0].click();", next_button)
                            time.sleep(0.7)

                            # Обрабатываем вновь появившиеся слайды
                            new_slides = driver.find_elements(By.CSS_SELECTOR,
                                                              "li.swiper-slide.j-product-photo:not([style*='display: none'])")

                            if not new_slides:
                                new_slides = driver.find_elements(By.CSS_SELECTOR, "div.sw-slider-product__item")

                            for slide in new_slides:
                                try:
                                    img_element = slide.find_element(By.CSS_SELECTOR, "img")
                                    img_url = extract_and_convert_image_url(img_element)
                                    if img_url:
                                        image_urls.add(img_url)
                                except:
                                    continue

                    except Exception as e:
                        logger.debug(f"Не удалось найти или нажать кнопку Next: {e}")

            elif current_strategy == "alternate_gallery" or current_strategy == "fallback":
                # Находим все элементы изображений
                img_elements = driver.find_elements(By.CSS_SELECTOR,
                                                    "img[src*='/catalog/'], img[data-src*='/catalog/'], img[src*='wbstatic']")

                for img in img_elements:
                    img_url = extract_and_convert_image_url(img)
                    if img_url:
                        image_urls.add(img_url)

        except Exception as e:
            logger.error(f"Критическая ошибка при извлечении изображений: {e}")

        logger.info(f"Всего найдено {len(image_urls)} уникальных URL изображений")
        return list(image_urls)