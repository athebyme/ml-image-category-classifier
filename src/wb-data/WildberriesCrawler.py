import random
import time
import os
import json
import base64
import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from typing import Dict, List
from loguru import logger

# Настройка логирования: пишем в консоль и в файл
logger.remove()
logger.add("crawler.log", format="{time} {level} {message}", level="INFO", rotation="5 MB")


class WildberriesCrawler:
    def __init__(self, category_targets: Dict[str, int], max_workers: int = 8):
        self.category_targets = category_targets
        self.max_workers = max_workers
        self.output_dir = "./output"
        self.products_queue = Queue()
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("Инициализация WildberriesCrawler завершена.")

    def get_driver(self):
        ua = UserAgent()
        options = webdriver.ChromeOptions()
        options.add_argument(f"user-agent={ua.random}")
        options.add_argument("--disable-application-cache")
        options.add_argument("--incognito")
        options.add_argument("--headless")  # Включаем headless режим для повышения производительности
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        # Отключаем загрузку изображений
        prefs = {"profile.managed_default_content_settings.images": 2}
        options.add_experimental_option("prefs", prefs)
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        logger.debug("Запущен новый драйвер Chrome в headless режиме.")
        return driver

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

    def wait_for_products_load(self, driver):
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "product-card__wrapper"))
            )
            logger.info("Карточки товаров успешно загружены.")
            return True
        except TimeoutException:
            logger.error("Timeout: карточки товаров не загрузились.")
            return False
        except Exception as e:
            logger.error(f"Ошибка при ожидании загрузки товаров: {e}")
            return False

    def download_image(self, url):
        try:
            response = requests.get(url, timeout=10)
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

    def process_product_page(self, driver, url, category):
        try:
            logger.info(f"Обработка карточки товара: {url}")
            driver.get(url)
            time.sleep(random.uniform(1, 2))  # уменьшенная задержка
            self.handle_age_verification(driver)

            wait = WebDriverWait(driver, 8)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "product-page__content")))

            soup = BeautifulSoup(driver.page_source, 'html.parser')

            name_element = soup.select_one("h1.product-page__title")
            name = name_element.text.strip() if name_element else ""

            description_element = soup.select_one("p.product-page__description")
            description = description_element.text.strip() if description_element else ""

            characteristics = {}
            chars_container = soup.select("table.product-params__table tr")
            for char in chars_container:
                name_elem = char.select_one("th.product-params__cell")
                value_elem = char.select_one("td.product-params__cell")
                if name_elem and value_elem:
                    characteristics[name_elem.text.strip()] = value_elem.text.strip()

            images = []
            image_elements = soup.select("div.slide__content img.photo-zoom__preview")
            for img in image_elements[:5]:
                src = img.get('src')
                if src:
                    if src.startswith('//'):
                        src = 'https:' + src
                    img_data = self.download_image(src)
                    if img_data:
                        images.append(img_data)

            article = self.extract_article_from_url(url)
            logger.info(f"Успешно обработан товар: {article} - {name}")
            return {
                "article": article,
                "name": name,
                "description": description,
                "characteristics": characteristics,
                "images": images,
                "category": category,
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
                    product_data = self.process_product_page(driver, url, category)
                    if product_data:
                        self.save_product(product_data)
                    self.products_queue.task_done()
                except Queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Ошибка в воркере: {e}")
                    self.products_queue.task_done()
        finally:
            driver.quit()
            logger.debug("Драйвер закрыт.")

    def collect_product_urls(self, category: str, target_count: int) -> List[str]:
        driver = self.get_driver()
        urls = []
        page = 1

        try:
            while len(urls) < target_count:
                search_url = f"https://www.wildberries.ru/catalog/0/search.aspx?search={category}&page={page}"
                driver.get(search_url)
                self.handle_age_verification(driver)

                if not self.wait_for_products_load(driver):
                    logger.error(f"Не удалось загрузить товары на странице {page} для категории {category}.")
                    break

                self.smooth_scroll(driver)

                soup = BeautifulSoup(driver.page_source, 'html.parser')
                product_links = soup.select("a.product-card__link.j-card-link.j-open-full-product-card")

                if not product_links:
                    logger.warning(f"Не найдены ссылки на товары на странице {page} для категории {category}.")
                    break

                for link in product_links:
                    href = link.get("href")
                    if href:
                        if not href.startswith('http'):
                            href = 'https://www.wildberries.ru' + href
                        urls.append(href)
                        if len(urls) >= target_count:
                            break

                logger.info(f"Страница {page}: собрано {len(urls)} ссылок для категории {category}.")
                if len(urls) >= target_count:
                    break

                page += 1
                time.sleep(random.uniform(1, 2))
        except Exception as e:
            logger.error(f"Ошибка при сборе ссылок для категории {category} на странице {page}: {e}")
        finally:
            driver.quit()
            logger.debug("Драйвер закрыт после сбора ссылок.")

        return urls[:target_count]

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
    crawler = WildberriesCrawler(category_targets_needed, max_workers=8)
    crawler.run()
