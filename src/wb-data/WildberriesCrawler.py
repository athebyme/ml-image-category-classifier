import logging
import random
import time
import os
import json
import base64
import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty

from selenium.webdriver import ActionChains, Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from typing import Dict, List
from loguru import logger

# Настройка логирования: пишем в консоль и в файл
logger.remove()
logger.add("crawler.log", format="{time} {level} {message}", level="INFO", rotation="5 MB")


class ExponentialBackoff:
    def __init__(self, initial_delay=5, max_delay=300, factor=2):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.factor = factor
        self.attempt = 0

    def reset(self):
        self.attempt = 0

    def delay(self):
        wait_time = min(self.initial_delay * (self.factor ** self.attempt), self.max_delay)
        self.attempt += 1
        jitter = random.uniform(0.8, 1.2)  # Add 20% jitter
        return wait_time * jitter



class WildberriesCrawler:
    def __init__(self, category_targets: Dict[str, int], max_workers: int = 8):
        self.category_targets = category_targets
        self.max_workers = max_workers
        self.output_dir = "./output"
        self.products_queue = Queue()
        self.session_count = 0
        self.max_requests_per_session = random.randint(15, 25)  # Randomize session length
        self.backoff = ExponentialBackoff()
        os.makedirs(self.output_dir, exist_ok=True)
        self.existing_articles = self.load_existing_articles()

        # Optional: Set up proxy list if available
        self.proxies = []  # Add your proxies here if available

        logger.info("Инициализация WildberriesCrawler завершена.")

    def get_driver(self):
        ua = UserAgent()
        options = webdriver.ChromeOptions()
        options.add_argument(f"user-agent={ua.random}")
        options.add_argument("--disable-application-cache")
        options.add_argument("--incognito")
        options.add_argument("--headless=new")  # Updated headless flag for newer Chrome
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-web-security")

        # Add proxy if available
        if hasattr(self, 'proxies') and self.proxies:
            proxy = random.choice(self.proxies)
            options.add_argument(f'--proxy-server={proxy}')
            logger.debug(f"Используем прокси: {proxy}")

        try:
            # Use system-installed ChromeDriver directly
            driver = webdriver.Chrome(options=options)
            logger.debug("Запущен новый драйвер Chrome в headless режиме.")
            driver.delete_all_cookies()
            logger.debug("Куки очищены успешно")
            return driver
        except Exception as e:
            logger.error(f"Ошибка при инициализации ChromeDriver: {e}")

            # Fallback to using WebDriverManager
            try:
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=options)
                driver.delete_all_cookies()
                return driver
            except Exception as e2:
                logger.error(f"Альтернативная инициализация тоже не удалась: {e2}")

                # Last resort - try finding chromedriver in specific locations
                try:
                    import shutil
                    chromedriver_path = shutil.which('chromedriver')
                    if chromedriver_path:
                        service = Service(chromedriver_path)
                        driver = webdriver.Chrome(service=service, options=options)
                        driver.delete_all_cookies()
                        return driver
                except Exception as e3:
                    logger.error(f"Последняя попытка тоже не удалась: {e3}")
                    raise

    def handle_age_verification(self, driver):
        try:
            age_button = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "button.confirm-age"))
            )
            age_button.click()
            time.sleep(0.5)  # уменьшенная задержка
            logger.debug("Возрастное подтверждение пройдено.")
        except TimeoutException:
            logger.debug("Возрастное подтверждение не найдено, продолжаем.")
            pass

    def smooth_scroll(self, driver, scroll_pause_time=0.5, scroll_increment=50, max_attempts_without_new=5):
        current_position = 0
        attempts_without_new = 0
        last_height = driver.execute_script("return document.body.scrollHeight")

        while attempts_without_new < max_attempts_without_new:
            driver.execute_script("window.scrollTo(0, arguments[0]);", current_position)
            time.sleep(0.02)  # быстрая прокрутка
            current_position += scroll_increment
            new_height = driver.execute_script("return document.body.scrollHeight")

            if current_position >= new_height:
                time.sleep(scroll_pause_time)
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    attempts_without_new += 1
                    logger.debug(
                        f"Новых данных не подгрузилось, попытка {attempts_without_new}/{max_attempts_without_new}")
                else:
                    attempts_without_new = 0
                    last_height = new_height
                current_position = new_height
        logger.info("Прокрутка страницы завершена.")

    def download_image(self, url):
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                logger.debug(f"Изображение успешно загружено: {url}")
                return base64.b64encode(response.content).decode('utf-8')
        except requests.exceptions.Timeout:
            logger.error(f"Timeout при загрузке изображения {url}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке изображения {url}: {e}")
        return None

    def extract_article_from_url(self, url):
        match = re.search(r'/catalog/(\d+)/detail\.aspx', url)
        if match:
            return match.group(1)
        return None
    def parse_wb_slider_images(self, driver, logger=None):
        """
        Парсит все изображения из слайдера Wildberries.

        Args:
            driver: экземпляр WebDriver
            logger: опциональный logger для записи ошибок

        Returns:
            list: список URL изображений в высоком качестве
        """
        if logger is None:
            logger = logging.getLogger(__name__)

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
                        # значит мы достигли конца слайдера
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

    def parse_wb_slider_images_high_res_v2(self, driver, logger=None):
        """
        Парсит все изображения из слайдера Wildberries в высоком разрешении.
        Сначала пытается получить обычные URL, затем использует canvas для получения data URL.

        Args:
            driver: экземпляр WebDriver
            logger: опциональный logger для записи ошибок

        Returns:
            list: список URL изображений или data URLs
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        image_urls = set()
        max_attempts = 20
        attempts = 0

        try:
            # Ждем загрузки основного слайдера
            slider = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "ul.swiper-wrapper"))
            )

            while attempts < max_attempts:
                # Получаем все видимые слайды на текущей странице
                slides = driver.find_elements(By.CSS_SELECTOR,
                                              "li.swiper-slide.j-product-photo:not([style*='display: none'])")

                images_before = len(image_urls)

                for slide in slides:
                    try:
                        # Сначала проверим, есть ли обычное изображение в слайде
                        try:
                            img_element = slide.find_element(By.CSS_SELECTOR, "img")
                            img_src = img_element.get_attribute("src")
                            if img_src and not img_src.startswith("data:"):
                                # Преобразуем URL в высокое разрешение
                                high_res_url = img_src.replace("/tm/", "/big/")
                                high_res_url = high_res_url.replace("/c246x328/", "/big/")
                                image_urls.add(high_res_url)
                                logger.debug(f"Добавлено обычное изображение: {high_res_url}")
                                continue  # Переходим к следующему слайду, если получили обычный URL
                        except Exception as img_e:
                            logger.debug(f"Не удалось получить обычное изображение: {str(img_e)}")
                            # Продолжаем к методу получения через canvas

                        # Кликаем по слайду для увеличения
                        driver.execute_script(
                            "arguments[0].querySelector('.slide__content').click();",
                            slide
                        )

                        # Ждем пока появится canvas zoom и получаем data URL
                        try:
                            zoom_canvas = WebDriverWait(driver, 5).until(
                                EC.presence_of_element_located(
                                    (By.CSS_SELECTOR, "canvas.photo-zoom__preview.j-image-canvas"))
                            )
                        except TimeoutException:
                            logger.debug("Canvas element не найден после клика.")
                            ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                            continue  # Переходим к следующему слайду

                        # Получаем data URL из canvas через JavaScript
                        data_url = driver.execute_script(
                            "return arguments[0].toDataURL('image/png');",
                            zoom_canvas
                        )

                        if data_url and data_url.startswith('data:image/png;base64,'):
                            # Добавляем data URL в нашу коллекцию
                            image_urls.add(data_url)
                            logger.debug(f"Добавлено изображение из canvas (data URL): {data_url[:50]}...")
                        else:
                            logger.warning("Не удалось получить data URL из canvas или неверный формат.")

                        # Закрываем zoom view
                        ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                        time.sleep(0.3)  # Увеличим паузу для более надежного закрытия зума

                    except Exception as e:
                        logger.debug(f"Ошибка при обработке слайда: {str(e)}")
                        # Убедимся, что зум закрыт перед переходом к следующему слайду
                        try:
                            ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                        except:
                            pass
                        continue

                # Если новых изображений нет, пробуем пролистнуть или завершаем
                if len(image_urls) == images_before:
                    try:
                        next_button = WebDriverWait(driver, 2).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "button.swiper-button-next"))
                        )

                        if 'swiper-button-disabled' in next_button.get_attribute('class'):
                            logger.debug("Кнопка Next заблокирована, завершаем.")
                            break

                        driver.execute_script("arguments[0].click();", next_button)
                        time.sleep(0.5)

                        # Прерываем если 2 попытки не дали новых изображений после пролистывания
                        if attempts > 1:
                            logger.debug("Больше нет новых изображений после пролистывания, завершаем.")
                            break

                    except TimeoutException:
                        logger.debug("Кнопка Next не найдена, возможно, все изображения показаны.")
                        break

                attempts += 1

        except Exception as e:
            logger.error(f"Критическая ошибка парсера изображений: {str(e)}")

        return list(image_urls)

    # Add this method if you want to save data URLs as files
    def save_data_url_to_file(self, data_url, product_id):
        """
        Сохраняет data URL как файл и возвращает путь к файлу
        """
        try:
            import base64
            import os
            from datetime import datetime

            # Create directory if it doesn't exist
            img_dir = os.path.join("images", str(product_id))
            os.makedirs(img_dir, exist_ok=True)

            # Extract the base64 encoded data
            header, encoded = data_url.split(",", 1)
            data = base64.b64decode(encoded)

            # Generate a unique filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            file_path = os.path.join(img_dir, f"image_{timestamp}.png")

            # Write to file
            with open(file_path, "wb") as f:
                f.write(data)

            return file_path
        except Exception as e:
            logging.error(f"Ошибка при сохранении data URL: {str(e)}")
            return None

    def process_product_page(self, driver, url, category):
        try:
            logger.info(f"Обработка карточки товара: {url}")
            driver.get(url)
            time.sleep(random.uniform(1, 4))
            self.handle_age_verification(driver)

            # Initialize these variables early to avoid reference errors
            # if the detail popup section fails
            description = ""
            characteristics = {}

            # Ждем загрузку основной информации о товаре
            WebDriverWait(driver, 20).until(
                EC.visibility_of_element_located((By.CLASS_NAME, "product-page__title"))
            )
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ul.breadcrumbs__list li.breadcrumbs__item"))
                )
            except TimeoutException:
                logger.warning(f"Timeout waiting for breadcrumbs on {url}")

            # Извлекаем базовую информацию: название, бренд, навигацию и начальные изображения
            main_soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Название товара
            name_element = main_soup.select_one("h1.product-page__title")
            name = name_element.text.strip() if name_element else ""

            # Бренд товара
            brand_element = main_soup.select_one("a.product-page__header-brand")
            brand = brand_element.text.strip() if brand_element else ""

            # Хлебные крошки (навигация)
            breadcrumb_elements = main_soup.select("ul.breadcrumbs__list li.breadcrumbs__item span[itemprop='name']")

            # If first attempt fails, try alternative selectors
            if not breadcrumb_elements:
                breadcrumb_elements = main_soup.select("ul.breadcrumbs__list li.breadcrumbs__item")
                breadcrumbs = [crumb.text.strip() for crumb in breadcrumb_elements if crumb.text.strip()]
                logger.info(f"Used alternative breadcrumb selector, found {len(breadcrumbs)} items")
            else:
                breadcrumbs = [crumb.text.strip() for crumb in breadcrumb_elements]
                logger.info(f"Used original breadcrumb selector, found {len(breadcrumbs)} items")

            # Первоначальное извлечение изображений (низкого качества, если есть)
            images = []
            for img in main_soup.select("div.slide__content img.photo-zoom__preview"):
                src = img.get('src')
                if src:
                    if src.startswith('//'):
                        src = 'https:' + src
                    images.append(src)

            try:
                details_btn = WebDriverWait(driver, 20).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button.product-page__btn-detail"))
                )
                driver.execute_script("arguments[0].scrollIntoView(true);", details_btn)
                time.sleep(0.5)

                try:
                    details_btn.click()
                except Exception as e:
                    driver.execute_script("arguments[0].click();", details_btn)

                # Ждем загрузки попапа
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "table.product-params__table"))
                )
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "section.product-details__description p.option__text"))
                )

                # Даем время на полную загрузку попапа
                time.sleep(1)

                # Парсим содержимое попапа
                popup_soup = BeautifulSoup(driver.page_source, 'html.parser')

                # Характеристики товара
                characteristics = {}
                for row in popup_soup.select("table.product-params__table tr"):
                    key_elem = row.select_one("th.product-params__cell")
                    val_elem = row.select_one("td.product-params__cell")
                    if key_elem and val_elem:
                        characteristics[key_elem.text.strip()] = val_elem.text.strip()

                # Описание товара - ищем по заголовку "Описание"
                description = ""
                description_section = popup_soup.find("section", class_="product-details__description")
                if description_section:
                    description_p = description_section.find("p", class_="option__text")
                    if description_p:
                        description = description_p.text.strip()
                    else:
                        logger.debug("Параграф с описанием не найден")
                else:
                    logger.debug("Секция с описанием не найдена")

                # Закрываем попап
                try:
                    close_btn = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "a.j-close.popup__close.close"))
                    )
                    close_btn.click()
                except Exception as e:
                    logger.debug(f"Не удалось закрыть попап: {e}")

            except Exception as e:
                logger.error(f"Ошибка при работе с попапом характеристик: {e}")

            images = self.parse_wb_slider_images_high_res_v2(driver, logger)

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
                "product-category-description": category,  # Изменено с "category" на "product-category-description"
                "url": url
            }
        except Exception as e:
            logger.error(f"Ошибка обработки карточки товара {url}: {e}")
            return None

    def worker(self):
        driver = self.get_driver()
        try:
            while True:
                try:
                    url, category = self.products_queue.get_nowait()
                    try:
                        product_data = self.process_product_page(driver, url, category)
                        if product_data:
                            self.save_product(product_data)
                    except Exception as e:
                        # Улучшенное логирование ошибок с трассировкой
                        logger.error(f"Ошибка при обработке URL {url}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                    finally:
                        self.products_queue.task_done()
                except Empty:
                    break
                except Exception as e:
                    logger.error(f"Неожиданная ошибка в воркере: {e}")
                    # Если задача не была взята, мы не должны отмечать ее выполненной
        finally:
            driver.quit()
            logger.debug("Драйвер закрыт.")

    def process_product_data(self, product_data, images):
        """
        Обрабатывает данные продукта и изображения

        Args:
            product_data: словарь с данными продукта
            images: список URL изображений или data URLs

        Returns:
            dict: обновленный словарь с данными продукта
        """
        # Добавляем изображения к данным продукта
        product_data['images'] = []

        for img in images:
            if img.startswith('data:image/'):
                # Option 1: Save data URL to file and add the file path
                # file_path = self.save_data_url_to_file(img, product_data.get('article', 'unknown'))
                # product_data['images'].append(file_path)

                # Option 2: Just add the data URL directly (note: this can make the JSON very large)
                product_data['images'].append(img)
            else:
                # For regular URLs, just add them directly
                product_data['images'].append(img)

        return product_data

    def load_existing_articles(self):
        """Загружает артикулы из уже собранных JSON файлов."""
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

    def wait_for_products_load(self, driver):
        try:
            # First try the original selector
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, "product-card__wrapper"))
                )
                logger.info("Карточки товаров успешно загружены (основной селектор).")
                return True
            except TimeoutException:
                logger.warning("Основной селектор не сработал, пробуем альтернативные...")

                # Try alternative selectors
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".product-card"))
                    )
                    logger.info("Карточки товаров успешно загружены (альтернативный селектор 1).")
                    return True
                except TimeoutException:
                    try:
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.j-card-link"))
                        )
                        logger.info("Карточки товаров успешно загружены (альтернативный селектор 2).")
                        return True
                    except TimeoutException:
                        logger.error("Не удалось найти товары на странице по всем селекторам.")

                        # Take screenshot for debugging
                        timestamp = int(time.time())
                        screenshot_path = f"error_page_{timestamp}.png"
                        driver.save_screenshot(screenshot_path)
                        logger.info(f"Сохранен скриншот проблемной страницы: {screenshot_path}")

                        # Get page source for debugging
                        with open(f"error_page_{timestamp}.html", "w", encoding="utf-8") as f:
                            f.write(driver.page_source)
                        logger.info(f"Сохранен HTML проблемной страницы: error_page_{timestamp}.html")

                        return False
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при ожидании загрузки товаров: {e}")
            return False

    def collect_product_urls(self, category: str, target_count: int) -> List[str]:
        urls = []
        page = 1
        max_retries = 3
        retry_count = 0
        max_consecutive_failures = 3
        consecutive_failures = 0
        collected_articles_for_category = set() # Для отслеживания артикулов, собранных в текущей категории

        while len(urls) < target_count and consecutive_failures < max_consecutive_failures:
            try:
                # Create a new driver for each page or after several retries
                driver = self.get_driver()

                try:
                    search_url = f"https://www.wildberries.ru/catalog/0/search.aspx?search={category}&page={page}"
                    logger.info(f"Загрузка страницы {page} для категории '{category}'")

                    driver.get(search_url)
                    time.sleep(random.uniform(3, 7))  # Random delay

                    self.handle_age_verification(driver)
                    self.simulate_human_behavior(driver)


                    if not self.wait_for_products_load(driver):
                        consecutive_failures += 1
                        logger.warning(
                            f"Не удалось загрузить товары на странице {page} (попытка {consecutive_failures}/{max_consecutive_failures})")
                        if consecutive_failures >= max_consecutive_failures:
                            logger.error(
                                f"Достигнут лимит последовательных неудач. Возможно, бот обнаружен. Переключаемся на следующую категорию.")
                            break
                        continue

                    consecutive_failures = 0

                    self.smooth_scroll(driver, scroll_pause_time=random.uniform(0.7, 1.5),
                                       scroll_increment=random.randint(40, 60))

                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    product_links = soup.select("a.product-card__link.j-card-link.j-open-full-product-card")

                    if not product_links:
                        logger.warning(f"Не найдены ссылки на товары на странице {page}. Пробуем другой селектор.")
                        product_links = soup.select("a.product-card__main.j-card-link")
                        if not product_links:
                            logger.warning(
                                f"Альтернативный селектор тоже не нашел товары. Попробуем перезагрузить страницу.")
                            retry_count += 1
                            if retry_count >= max_retries:
                                logger.error(f"Достигнут лимит попыток для страницы {page}. Переходим к следующей.")
                                page += 1
                                retry_count = 0
                            continue

                    retry_count = 0

                    new_urls_count = 0
                    for link in product_links:
                        href = link.get("href")
                        if href:
                            if not href.startswith('http'):
                                href = 'https://www.wildberries.ru' + href

                            # Извлекаем артикул из URL (пример: .../catalog/221596740/detail.aspx)
                            article_match = re.search(r'/catalog/(\d+)/detail\.aspx', href)
                            article = article_match.group(1) if article_match else None

                            if article:
                                if article not in self.existing_articles and article not in collected_articles_for_category: # Проверяем, что артикула нет в уже собранных и в текущей сессии
                                    urls.append(href)
                                    collected_articles_for_category.add(article) # Добавляем в собранные в текущей сессии
                                    new_urls_count += 1
                                    if len(urls) >= target_count:
                                        break
                                else:
                                    logger.debug(f"Артикул {article} уже собран или в списке существующих. Пропускаем.")
                            else:
                                logger.warning(f"Не удалось извлечь артикул из URL: {href}. URL будет добавлен, но проверка на дубликат по артикулу невозможна.")
                                if href not in urls: # Проверка на дубликат по URL на всякий случай
                                    urls.append(href)
                                    new_urls_count += 1
                                    if len(urls) >= target_count:
                                        break


                    logger.info(
                        f"Страница {page}: добавлено {new_urls_count} новых ссылок (всего: {len(urls)}/{target_count})")

                    if new_urls_count == 0:
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            logger.warning(
                                f"Слишком много страниц без новых товаров. Возможно, достигнут конец каталога.")
                            break
                    else:
                        consecutive_failures = 0

                    page += 1
                    time.sleep(random.uniform(2, 5))  # Random delay between pages


                finally:
                    driver.quit()
                    logger.debug("Драйвер закрыт после обработки страницы.")


            except Exception as e:
                logger.error(f"Ошибка при сборе ссылок на странице {page}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                consecutive_failures += 1
                time.sleep(5)  # Wait after error


        return urls[:target_count]

    def simulate_human_behavior(self, driver):
        """Симулирует поведение человека для обхода обнаружения бота"""
        try:
            # Случайное движение мыши
            viewport_width = driver.execute_script("return window.innerWidth")
            viewport_height = driver.execute_script("return window.innerHeight")

            # Перемещение к случайной точке на странице
            action = ActionChains(driver)

            # 2-3 случайных движения мыши
            for _ in range(random.randint(2, 3)):
                x = random.randint(10, viewport_width - 10)
                y = random.randint(10, viewport_height - 10)
                action.move_by_offset(x, y)
                time.sleep(random.uniform(0.1, 0.3))

            # Иногда кликаем на неинтерактивный элемент (например, пустое пространство)
            if random.random() < 0.3:  # 30% chance
                action.click()

            action.perform()

            # Иногда прокручиваем страницу немного вниз и обратно
            if random.random() < 0.4:  # 40% chance
                scroll_amount = random.randint(100, 300)
                driver.execute_script(f"window.scrollBy(0, {scroll_amount})")
                time.sleep(random.uniform(0.5, 1.2))
                driver.execute_script(f"window.scrollBy(0, {-scroll_amount})")

        except Exception as e:
            logger.debug(f"Ошибка при симуляции поведения человека: {e}")

    def randomize_browser_fingerprint(self, options):
        """Randomizes browser fingerprint settings to avoid detection"""
        # Random screen dimensions
        width = random.choice([1366, 1440, 1920, 2560])
        height = random.choice([768, 900, 1080, 1440])
        options.add_argument(f"--window-size={width},{height}")

        # Random language
        languages = ["en-US,en;q=0.9", "en-GB,en;q=0.9", "ru-RU,ru;q=0.9", "de-DE,de;q=0.9"]
        options.add_argument(f"--lang={random.choice(languages)}")

        # Change accept header
        options.add_argument("--accept-lang=en-US,en;q=0.9,ru;q=0.8")

        return options

    def save_product(self, product_data):
        if product_data and product_data.get('article'):
            article = product_data['article']
            filename = f"product_{article}.json"
            filepath = os.path.join(self.output_dir, filename)
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(product_data, f, ensure_ascii=False, indent=2)
                logger.success(f"Сохранён товар {article} в {filepath}")
            except Exception as e:
                logger.error(f"Ошибка сохранения данных товара {article} в {filepath}: {e}")
        else:
            logger.warning("Данные товара или номер статьи отсутствуют. Сохранение пропущено.")

    def run(self):
        logger.info("Запуск краулера Wildberries...")
        start_time = time.time()
        for category, target_count in self.category_targets.items():
            logger.info(f"Сбор ссылок для категории: {category} (цель: {target_count} товаров)")
            product_urls = self.collect_product_urls(category, target_count)
            logger.info(f"Для категории {category} собрано {len(product_urls)} ссылок.")

            for url in product_urls:
                self.products_queue.put((url, category))

            logger.info(f"Начало обработки товаров категории: {category} с использованием {self.max_workers} воркеров.")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                workers = [executor.submit(self.worker) for _ in range(self.max_workers)]
                for worker in as_completed(workers):
                    worker.result()

            logger.info(f"Обработка категории {category} завершена.")

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Краулинг завершён за {elapsed_time:.2f} секунд.")
        logger.info(f"Все товары сохранены в каталоге '{self.output_dir}'.")


if __name__ == "__main__":
    category_targets_needed = {
        "Презервативы": 4000 - 700,
        "Вибраторы": 4000 - 650,
        "Фаллоимитаторы": 4000 - 600,
        "Анальные пробки": 4000 - 590,
        "Мастурбаторы мужские": 4000 - 470,
        "Комплекты эротик": 4000 - 340,
        "Ролевые костюмы эротик": 4000 - 330,
        "Платья эротик": 4000 - 320,
        "Куклы секс": 4000 - 270,
        "Наборы игрушек для взрослых": 4000 - 260,
        "Вакуумные помпы эротик": 4000 - 240,
        "Чулки эротик": 4000 - 220,
        "Костюмы эротик": 4000 - 210,
        "Лубриканты": 4000 - 200,
        "Насадки на член": 4000 - 190,
        "Анальные шарики": 4000 - 180,
        "Вибропули": 4000 - 170,
        "Массажные средства эротик": 4000 - 160,
        "Анальные бусы": 4000 - 150,
        "Вакуумно-волновые стимуляторы": 4000 - 140,
        "Маски эротик": 4000 - 130,
        "Возбуждающие средства": 4000 - 120,
        "Бандажи эротик": 4000 - 110,
        "Пэстис эротик": 4000 - 100,
        "Трусы эротик": 4000 - 90,
        "Свечи эротик": 4000 - 85,
        "Секс машины": 4000 - 80,
        "Пульсаторы": 4000 - 75,
        "Пояса эротик": 4000 - 70,
        "Пеньюары эротик": 4000 - 65,
        "БДСМ комплекты": 4000 - 60,
        "Благовония эротик": 4000 - 55,
        "Боди эротик": 4000 - 50,
        "Браслеты эротик": 4000 - 48,
        "Бюстгальтеры эротик": 4000 - 46,
        "Вагинальные тренажеры": 4000 - 44,
        "Вагинальные шарики": 4000 - 42,
        "Веревки для бондажа": 4000 - 40,
        "Виброяйца": 4000 - 38,
        "Возбуждающие напитки": 4000 - 36,
        "Галстуки эротик": 4000 - 34,
        "Гартеры эротик": 4000 - 32,
        "Гидропомпы эротик": 4000 - 30,
        "Зажимы для сосков": 4000 - 28,
        "Имитаторы груди": 4000 - 26,
        "Календари эротические": 4000 - 24,
        "Кляпы эротик": 4000 - 22,
        "Колготки эротик": 4000 - 20,
        "Корсеты эротик": 4000 - 18,
        "Леггинсы эротик": 4000 - 16,
        "Манжеты эротик": 4000 - 14,
        "Наручники эротик": 4000 - 12,
        "Насадки для вибраторов": 4000 - 10,
        "Насадки на страпон": 4000 - 8,
        "Оковы эротик": 4000 - 6,
        "Ошейники эротик": 4000 - 4,
        "Перчатки эротик": 4000 - 2,
        "Плетки эротик": 4000 - 1,
        "Поводки эротик": 4000 - 1,
        "Подвязки эротик": 4000 - 1,
        "Портупеи эротик": 4000 - 1,
        "Пояса верности": 4000 - 1,
        "Простыни БДСМ": 4000 - 1,
        "Секс качели": 4000 - 1,
        "Секс мячи": 4000 - 1,
        "Топы эротик": 4000 - 1,
        "Трусы для страпона": 4000 - 1,
        "Увеличители члена": 4000 - 1,
        "Утяжелители эротик": 4000 - 1,
        "Фаллопротезы": 4000 - 1,
        "Чокеры эротик": 4000 - 1,
        "Шорты эротик": 4000 - 1,
        "Электростимуляторы": 4000 - 1,
        "Эрекционные кольца": 4000 - 1,
        "Юбки эротик": 4000 - 1,
    }
    output_dir = "./output"
    existing_category_counts = {}

    # 1. Прочитать JSON файлы из ./output и подсчитать категории (как и раньше)
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(output_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        category = data.get("product-category-description")
                        if category:
                            existing_category_counts[category] = existing_category_counts.get(category, 0) + 1
                except json.JSONDecodeError:
                    print(f"Ошибка декодирования JSON в файле: {filename}")
                except Exception as e:
                    print(f"Ошибка при чтении файла {filename}: {e}")

    print("Существующие товары по категориям:")
    for category, count in existing_category_counts.items():
        print(f'"{category}": {count},')

    # 2. Вычислить ИСХОДНЫЕ целевые суммы (до вычитания собранных)
    initial_targets = category_targets_needed.copy()  # Копия для сохранения исходных значений
    # for category in initial_targets:
    #     initial_targets[category] = 4000  # Предполагаем, что 4000 - это "базовое" значение для масштабирования

    current_total_initial_target = sum(initial_targets.values())  # Суммируем ИСХОДНЫЕ значения
    print(f"\nИсходная общая сумма (для масштабирования): {current_total_initial_target}")

    # 3. Определить коэффициент масштабирования (используя ИСХОДНУЮ сумму)
    desired_total_target = current_total_initial_target / 4
    scaling_factor = desired_total_target / current_total_initial_target
    print(f"Коэффициент масштабирования: {scaling_factor}")

    # 4. Масштабировать ИСХОДНЫЕ целевые значения, а затем вычесть собранные
    category_targets_needed_scaled = {}
    for category, initial_target in initial_targets.items():  # Итерируем по ИСХОДНЫМ целям
        scaled_initial_target = round(initial_target * scaling_factor)  # Масштабируем ИСХОДНОЕ значение
        existing_count = existing_category_counts.get(category, 0)
        adjusted_target = max(0,
                              scaled_initial_target - existing_count)  # Вычитаем собранные из МАСШТАБИРОВАННОГО значения
        category_targets_needed_scaled[category] = adjusted_target

    # 5. Вывести итоговые значения
    print("\nИтоговые целевые значения (после масштабирования и учета существующих):")
    for category, target in category_targets_needed_scaled.items():
        print(f'"{category}": {target},')

    new_total_target = sum(category_targets_needed_scaled.values())
    print(f"\nНовая общая сумма: {new_total_target}")

    # Замените original словарь на scaled словарь в вашем коде
    category_targets_needed = category_targets_needed_scaled

    crawler = WildberriesCrawler(category_targets_needed_scaled, max_workers=4)
    crawler.run()
