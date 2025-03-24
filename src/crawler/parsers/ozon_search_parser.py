"""
Модуль для парсинга страниц поиска и категорий Ozon.
"""
import re
import time
import random
import json
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from ..logging_setup import logger


class OzonSearchParser:
    """
    Класс для парсинга страниц поиска Ozon и сбора ссылок на товары.
    """

    def __init__(self, browser_manager, captcha_solver=None):
        """
        Инициализирует парсер страниц поиска Ozon.

        Args:
            browser_manager (BrowserManager): Менеджер браузера.
            captcha_solver (CaptchaSolver, optional): Решатель CAPTCHA.
        """
        self.browser_manager = browser_manager
        self.captcha_solver = captcha_solver
        self.existing_articles = set()
        self.search_base_url = "https://www.ozon.ru/search/?text={}&from_global=true&page={}"
        self.max_pages_per_term = 40
        self.max_consecutive_failures = 3
        self.circuit_breaker = {
            "failures": 0,
            "threshold": 5,
            "is_open": False,
            "reset_after": 180  # секунды
        }
        self.circuit_open_time = None

    def set_existing_articles(self, articles):
        """
        Устанавливает список существующих артикулов.

        Args:
            articles (set): Набор артикулов.
        """
        self.existing_articles = articles

    def wait_for_products_load(self, driver):
        """
        Ожидает загрузки товаров на странице поиска Ozon.

        Args:
            driver (WebDriver): Экземпляр веб-драйвера.

        Returns:
            bool: True, если товары загрузились успешно, иначе False.
        """
        try:
            # Сначала убеждаемся, что страница полностью загружена
            WebDriverWait(driver, 30).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )

            # Добавляем случайные задержки для имитации человеческого поведения
            time.sleep(random.uniform(2, 5))

            # Пробуем несколько селекторов по порядку для Ozon
            selectors = [
                "div.iq8",  # Основной селектор карточек товаров Ozon
                "div.iy3",
                "div[data-widget='searchResultsV2']",
                "div.tile-hover-target",
                "div.ju"
            ]

            for selector in selectors:
                try:
                    WebDriverWait(driver, 8).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
                    )
                    logger.info(f"Товары Ozon успешно загружены с использованием селектора: {selector}")
                    return True
                except TimeoutException:
                    logger.debug(f"Селектор {selector} не сработал для Ozon, пробуем следующий")
                    continue

            # Проверяем сообщение "ничего не найдено"
            try:
                no_results = driver.find_element(By.XPATH, "//h1[contains(text(), 'не наш')]")
                if no_results:
                    logger.info("Ozon вернул 'ничего не найдено'")
                    return False
            except:
                pass

            # Если мы дошли до этого места, все селекторы не сработали
            logger.error("Все селекторы товаров Ozon не сработали")

            # Делаем скриншот для отладки
            timestamp = int(time.time())
            screenshot_path = f"error_ozon_page_{timestamp}.png"
            driver.save_screenshot(screenshot_path)
            logger.info(f"Сохранен скриншот ошибки Ozon: {screenshot_path}")

            return False
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при ожидании товаров Ozon: {e}")
            return False

    def get_alternative_search_terms(self, category):
        """
        Генерирует альтернативные поисковые запросы на основе категории для Ozon.

        Args:
            category (str): Исходная категория.

        Returns:
            list: Список альтернативных поисковых запросов.
        """
        # Базовый словарь альтернатив для разных категорий на Ozon
        alternatives = {
            "Презервативы": ["Презервативы", "Контрацептивы", "Средства контрацепции", "Durex", "Contex"],
            "Вибраторы": ["Вибратор для женщин", "Женский вибратор", "Массажер интимный", "Клиторальный вибратор"],
            "Фаллоимитаторы": ["Дилдо", "Секс игрушка для женщин", "Интимная игрушка", "Фаллос"]
            # Добавьте другие категории по необходимости
        }

        # Если для категории есть предопределенные альтернативы, используем их
        if category in alternatives:
            return [category] + alternatives[category]

        # Генерируем стандартные вариации
        result = [category]  # Исходная категория всегда первая

        # Удаляем "эротик" и другие суффиксы для Ozon
        suffix_words = ["эротик", "для женщин", "для мужчин", "для пар"]
        for suffix in suffix_words:
            if category.lower().endswith(f" {suffix}"):
                clean_category = category[:-len(suffix) - 1].strip()
                if clean_category and clean_category != category:
                    result.append(clean_category)
                    break

        # Удаляем спецсимволы
        clean_category = re.sub(r'[^\w\s]', '', category)
        if clean_category != category:
            result.append(clean_category)

        # Добавляем вариации во множественном/единственном числе
        if category.endswith("ы"):
            singular = category[:-1]
            result.append(singular)
        elif not category.endswith("ы") and not category.endswith("и"):
            plural = category + "ы"
            result.append(plural)

        # Для категорий с несколькими словами
        words = category.split()
        if len(words) > 1:
            # Основное слово отдельно
            main_word = words[0]
            if len(main_word) > 3:  # Проверка, чтобы не добавлять предлоги
                result.append(main_word)

            # Перестановки слов
            for i in range(1, len(words)):
                rotated = ' '.join(words[i:] + words[:i])
                if rotated != category:
                    result.append(rotated)

        # Для товаров 18+ добавляем эвфемизмы и маркетинговые термины
        ozon_terms = ["массажер", "товары для взрослых", "товары для здоровья", "интимные товары"]
        if any(word in category.lower() for word in ["эротик", "секс", "бдсм", "анальн"]):
            for term in ozon_terms:
                if term not in category.lower():
                    result.append(f"{category} {term}")

        # Удаляем дубликаты, сохраняя порядок
        unique_results = []
        seen = set()
        for term in result:
            if term.lower() not in seen:
                unique_results.append(term)
                seen.add(term.lower())

        return unique_results

    def try_switch_to_next_search_term(self, alternative_terms, current_idx):
        """
        Проверяет, можно ли переключиться на следующий поисковый термин для Ozon.

        Args:
            alternative_terms (list): Список альтернативных поисковых запросов.
            current_idx (int): Текущий индекс поискового запроса.

        Returns:
            bool: True, если можно переключиться на следующий запрос, иначе False.
        """
        return current_idx < len(alternative_terms) - 1

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

    def collect_product_urls(self, category, target_count):
        """
        Собирает URL-адреса товаров для указанной категории на Ozon.

        Args:
            category (str): Категория товаров для поиска.
            target_count (int): Целевое количество URL для сбора.

        Returns:
            list: Список URL-адресов товаров.
        """
        urls = []
        page = 1
        max_retries = 3
        retry_count = 0
        consecutive_failures = 0
        collected_articles_for_category = set()
        base_delay = 5

        # Альтернативные поисковые запросы
        alternative_search_terms = self.get_alternative_search_terms(category)
        current_search_idx = 0
        current_search_term = category

        while len(urls) < target_count:
            # Проверка автоматического выключателя
            if self.circuit_breaker["is_open"]:
                current_time = time.time()
                if self.circuit_open_time and (
                        current_time - self.circuit_open_time > self.circuit_breaker["reset_after"]):
                    # Сбрасываем автоматический выключатель
                    self.circuit_breaker["is_open"] = False
                    self.circuit_breaker["failures"] = 0
                    logger.info("Автоматический выключатель Ozon сброшен после периода охлаждения")
                else:
                    logger.warning("Автоматический выключатель Ozon открыт, приостановка запросов")
                    time.sleep(30)  # Ждем, прежде чем проверить снова
                    continue

            # Адаптивная задержка в зависимости от частоты сбоев
            if consecutive_failures > 0:
                delay = base_delay * (1.5 ** consecutive_failures)
                logger.info(f"Увеличение задержки до {delay:.2f}с из-за сбоев Ozon")
                time.sleep(delay)

            # Проверка лимита страниц для текущего поискового запроса
            if page > self.max_pages_per_term:
                logger.warning(
                    f"Достигнут предел страниц ({self.max_pages_per_term}) для запроса Ozon: '{current_search_term}'")
                if current_search_idx < len(alternative_search_terms) - 1:
                    current_search_idx += 1
                    current_search_term = alternative_search_terms[current_search_idx]
                    page = 1
                    consecutive_failures = 0
                    logger.info(f"Переключаемся на альтернативный поисковый запрос Ozon: '{current_search_term}'")
                    continue
                else:
                    logger.warning(f"Исчерпаны все поисковые запросы Ozon. Завершаем с {len(urls)} товарами.")
                    break

            try:
                # Создаем новый драйвер для каждой страницы
                driver = self.browser_manager.get_driver()

                try:
                    # Формируем URL страницы поиска Ozon
                    search_url = self.search_base_url.format(current_search_term, page)
                    logger.info(f"Загрузка страницы {page} поиска Ozon для запроса '{current_search_term}'")

                    # Загружаем страницу
                    driver.get(search_url)
                    time.sleep(random.uniform(1, 3))  # Короткая пауза после первоначального запроса

                    # Проверяем наличие CAPTCHA
                    if self.captcha_solver and self.captcha_solver.detect_captcha(driver):
                        logger.warning(f"Обнаружена CAPTCHA на странице поиска Ozon")
                        if not self.captcha_solver.handle_captcha(driver):
                            logger.error("Не удалось обработать CAPTCHA Ozon, переключаемся на другой запрос")
                            consecutive_failures += 1
                            driver.quit()

                            if consecutive_failures >= self.max_consecutive_failures:
                                if self.try_switch_to_next_search_term(alternative_search_terms, current_search_idx):
                                    current_search_idx += 1
                                    current_search_term = alternative_search_terms[current_search_idx]
                                    page = 1
                                    consecutive_failures = 0
                                    logger.info(
                                        f"Переключаемся на альтернативный поисковый запрос Ozon: '{current_search_term}'")
                                else:
                                    logger.error(
                                        f"Исчерпаны все альтернативные поисковые запросы Ozon. Завершаем сбор ссылок.")
                                    break
                            continue

                    # Ждем загрузки товаров
                    if not self.wait_for_products_load(driver):
                        consecutive_failures += 1
                        logger.warning(
                            f"Не удалось загрузить товары Ozon на странице {page} (попытка {consecutive_failures}/{self.max_consecutive_failures})")

                        if consecutive_failures >= self.max_consecutive_failures:
                            if self.try_switch_to_next_search_term(alternative_search_terms, current_search_idx):
                                current_search_idx += 1
                                current_search_term = alternative_search_terms[current_search_idx]
                                page = 1
                                consecutive_failures = 0
                                logger.info(
                                    f"Переключаемся на альтернативный поисковый запрос Ozon: '{current_search_term}'")
                            else:
                                logger.error(
                                    f"Исчерпаны все альтернативные поисковые запросы Ozon. Завершаем сбор ссылок.")
                                break
                        driver.quit()
                        continue

                    consecutive_failures = 0
                    success = True

                    # Плавная прокрутка для загрузки всего контента
                    self.browser_manager.smooth_scroll(driver,
                                                       scroll_pause_time=random.uniform(0.7, 1.5),
                                                       scroll_increment=random.randint(40, 60))

                    # Парсим страницу
                    soup = BeautifulSoup(driver.page_source, 'html.parser')

                    # Извлекаем ссылки на товары
                    product_links = []

                    # Пробуем извлечь из JSON-данных (наиболее надежный метод для Ozon)
                    json_links = self.extract_product_urls_from_json(driver.page_source)
                    if json_links:
                        product_links = json_links

                    # Если не получилось через JSON, пробуем через DOM
                    if not product_links:
                        # Пробуем разные селекторы Ozon
                        selectors = [
                            "a.tile-hover-target",
                            "a[href*='/product/']",
                            "div.iq8 a",
                            "div.ju a"
                        ]

                        for selector in selectors:
                            links = soup.select(selector)
                            if links:
                                product_links = links
                                break

                    # Если ни один метод не сработал, пробуем с помощью Selenium
                    if not product_links:
                        try:
                            elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='/product/']")
                            product_links = [{"href": el.get_attribute("href")} for el in elements if
                                             el.get_attribute("href")]
                        except Exception as e:
                            logger.error(f"Ошибка при извлечении ссылок через Selenium: {e}")

                    if not product_links:
                        logger.warning(f"Не найдены ссылки на товары Ozon на странице {page}")
                        consecutive_failures += 1

                        if consecutive_failures >= self.max_consecutive_failures:
                            if self.try_switch_to_next_search_term(alternative_search_terms, current_search_idx):
                                current_search_idx += 1
                                current_search_term = alternative_search_terms[current_search_idx]
                                page = 1
                                consecutive_failures = 0
                                logger.info(
                                    f"Переключаемся на альтернативный поисковый запрос Ozon: '{current_search_term}'")
                                driver.quit()
                                continue
                            else:
                                logger.error(
                                    f"Исчерпаны все альтернативные поисковые запросы Ozon. Завершаем сбор ссылок.")
                                break

                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error(f"Достигнут лимит попыток для страницы Ozon {page}. Переходим к следующей.")
                            page += 1
                            retry_count = 0
                        driver.quit()
                        continue

                    retry_count = 0

                    # Обрабатываем найденные ссылки
                    new_urls_count = 0
                    for link in product_links:
                        href = link.get("href") if hasattr(link, "get") else link.get("href", "")

                        if href:
                            # Преобразуем относительные URL в абсолютные
                            if not href.startswith('http'):
                                href = 'https://www.ozon.ru' + href

                            # Проверяем, что это действительно ссылка на товар
                            if "/product/" in href or "/context/detail/id/" in href:
                                # Извлекаем артикул
                                article = self.extract_article_from_url(href)

                                # Проверяем, не собирали ли мы уже этот товар
                                if article:
                                    if article not in self.existing_articles and article not in collected_articles_for_category:
                                        urls.append(href)
                                        collected_articles_for_category.add(article)
                                        new_urls_count += 1
                                        if len(urls) >= target_count:
                                            break
                                    else:
                                        logger.debug(f"Артикул Ozon {article} уже собран. Пропускаем.")
                                else:
                                    # Если не удалось извлечь артикул, добавляем URL, если его еще нет
                                    logger.warning(f"Не удалось извлечь артикул из URL Ozon: {href}")
                                    if href not in urls:
                                        urls.append(href)
                                        new_urls_count += 1
                                        if len(urls) >= target_count:
                                            break

                    logger.info(
                        f"Страница Ozon {page}: добавлено {new_urls_count} новых ссылок (всего: {len(urls)}/{target_count})")

                    # Если на странице не нашли новых товаров, увеличиваем счетчик неудачных попыток
                    if new_urls_count == 0:
                        consecutive_failures += 1

                        if consecutive_failures >= self.max_consecutive_failures:
                            if self.try_switch_to_next_search_term(alternative_search_terms, current_search_idx):
                                current_search_idx += 1
                                current_search_term = alternative_search_terms[current_search_idx]
                                page = 1
                                consecutive_failures = 0
                                logger.info(
                                    f"Страницы без новых товаров Ozon. Переключаемся на '{current_search_term}'")
                            else:
                                logger.warning(
                                    f"Исчерпаны все альтернативные поисковые запросы Ozon. Завершаем с {len(urls)} товарами.")
                                break
                    else:
                        consecutive_failures = 0
                        success = True

                    # Обновляем автоматический выключатель
                    if not success:
                        self.circuit_breaker["failures"] += 1
                        if self.circuit_breaker["failures"] >= self.circuit_breaker["threshold"]:
                            self.circuit_breaker["is_open"] = True
                            self.circuit_open_time = time.time()
                            logger.warning("Автоматический выключатель Ozon открыт из-за повторяющихся сбоев")

                            # Переключаемся на другой запрос при срабатывании выключателя
                            if current_search_idx < len(alternative_search_terms) - 1:
                                current_search_idx += 1
                                current_search_term = alternative_search_terms[current_search_idx]
                                page = 1
                    else:
                        # Сбрасываем счетчик сбоев при успехе
                        self.circuit_breaker["failures"] = max(0, self.circuit_breaker["failures"] - 1)

                    page += 1

                    # Случайный пропуск страниц для имитации человеческого поведения
                    if random.random() < 0.2:  # 20% вероятность
                        skip_pages = random.randint(1, 3)
                        page += skip_pages
                        logger.info(f"Случайно пропускаем {skip_pages} страниц Ozon для естественности")

                    time.sleep(random.uniform(2, 5))  # Случайная задержка между страницами

                finally:
                    driver.quit()
                    logger.debug("Драйвер Ozon закрыт после обработки страницы.")

            except Exception as e:
                logger.error(f"Ошибка при сборе ссылок Ozon на странице {page}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                consecutive_failures += 1

                # Проверяем, похоже ли исключение на CAPTCHA или обнаружение бота
                if any(term in str(e).lower() for term in ["captcha", "challenge", "robot", "автоматизированными"]):
                    logger.warning("Обнаружена возможная проблема с защитой от ботов Ozon. Охлаждение...")
                    time.sleep(random.uniform(60, 120))  # Длительное охлаждение при подозрении на блокировку

                    # Пробуем переключиться на другие поисковые запросы при подозрении на блокировку
                    if current_search_idx < len(alternative_search_terms) - 1:
                        current_search_idx += 1
                        current_search_term = alternative_search_terms[current_search_idx]
                        page = 1
                        consecutive_failures = 0
                        logger.info(f"Подозрение на блокировку Ozon, переключаемся на: '{current_search_term}'")
                else:
                    time.sleep(5)  # Обычная пауза после других ошибок

        return urls[:target_count]

    def extract_product_urls_from_json(self, page_source):
        """
        Извлекает URL товаров из JSON-данных на странице поиска Ozon.

        Args:
            page_source (str): Исходный код страницы.

        Returns:
            list: Список объектов с URL товаров.
        """
        try:
            # Ищем JSON-данные с товарами
            json_pattern = r'<script type="application/json" id="state-searchResultsV2">(.+?)</script>'
            json_match = re.search(json_pattern, page_source, re.DOTALL)

            if not json_match:
                # Альтернативный паттерн
                json_pattern = r'<script type="application/json" id="__NEXT_DATA__">(.+?)</script>'
                json_match = re.search(json_pattern, page_source, re.DOTALL)

            if json_match:
                json_data = json_match.group(1)
                data = json.loads(json_data)

                # Ищем ссылки на товары в JSON
                product_links = []

                # Рекурсивно ищем ссылки в сложной структуре JSON Ozon
                product_links = self._find_product_urls_in_json(data)

                # Преобразуем ссылки в нужный формат
                formatted_links = []
                for link in product_links:
                    if isinstance(link, str):
                        formatted_links.append({"href": link})
                    elif isinstance(link, dict) and "href" in link:
                        formatted_links.append(link)

                return formatted_links

            return []
        except Exception as e:
            logger.error(f"Ошибка при извлечении ссылок на товары Ozon из JSON: {e}")
            return []

    def _find_product_urls_in_json(self, data, max_depth=5, current_depth=0):
        """
        Рекурсивно ищет URL товаров в JSON-структуре Ozon.

        Args:
            data: JSON-данные для поиска.
            max_depth (int): Максимальная глубина рекурсии.
            current_depth (int): Текущая глубина рекурсии.

        Returns:
            list: Список URL товаров.
        """
        if current_depth >= max_depth:
            return []

        product_urls = []

        if isinstance(data, dict):
            # Проверяем ключи, которые могут содержать URL товаров Ozon
            for key, value in data.items():
                if key == "link" and isinstance(value, str) and (
                        "/product/" in value or "/context/detail/id/" in value):
                    product_urls.append(value)

                # Проверяем также ключи "url", "href" и т.д.
                if key in ["url", "href"] and isinstance(value, str) and (
                        "/product/" in value or "/context/detail/id/" in value):
                    product_urls.append(value)

                # Специфические для Ozon ключи
                if key == "items" and isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            if "link" in item and isinstance(item["link"], str) and (
                                    "/product/" in item["link"] or "/context/detail/id/" in item["link"]):
                                product_urls.append(item["link"])
                            elif "action" in item and isinstance(item["action"], dict) and "link" in item["action"]:
                                link = item["action"]["link"]
                                if isinstance(link, str) and ("/product/" in link or "/context/detail/id/" in link):
                                    product_urls.append(link)

                # Рекурсивно ищем в значениях
                if isinstance(value, (dict, list)):
                    product_urls.extend(self._find_product_urls_in_json(value, max_depth, current_depth + 1))

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    product_urls.extend(self._find_product_urls_in_json(item, max_depth, current_depth + 1))

        return product_urls