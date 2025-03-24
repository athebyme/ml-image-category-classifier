"""
Модуль для парсинга страниц поиска Wildberries.
"""
import re
import time
import random
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from ..logging_setup import logger
from ..config import MAX_CONSECUTIVE_FAILURES, MAX_PAGES_PER_TERM


class WildberriesSearchParser:
    """
    Класс для парсинга страниц поиска Wildberries и сбора ссылок на товары.
    """

    def __init__(self, browser_manager, captcha_solver=None):
        """
        Инициализирует парсер страниц поиска.

        Args:
            browser_manager (BrowserManager): Менеджер браузера.
            captcha_solver (CaptchaSolver, optional): Решатель CAPTCHA.
        """
        self.browser_manager = browser_manager
        self.captcha_solver = captcha_solver
        self.existing_articles = set()
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
        Ожидает загрузки товаров на странице поиска.

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

            # Пробуем несколько селекторов по порядку с правильной обработкой ошибок
            selectors = [
                ".product-card__wrapper",
                ".product-card",
                "a.j-card-link",
                "[data-card-index]",
                ".catalog-page .product-card"
            ]

            for selector in selectors:
                try:
                    WebDriverWait(driver, 8).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
                    )
                    logger.info(f"Товары успешно загружены с использованием селектора: {selector}")
                    return True
                except TimeoutException:
                    logger.debug(f"Селектор {selector} не сработал, пробуем следующий")
                    continue

            # Если мы дошли до этого места, все селекторы не сработали
            logger.error("Все селекторы товаров не сработали")

            # Делаем скриншот для отладки
            timestamp = int(time.time())
            screenshot_path = f"error_page_{timestamp}.png"
            driver.save_screenshot(screenshot_path)
            logger.info(f"Сохранен скриншот ошибки: {screenshot_path}")

            return False
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при ожидании товаров: {e}")
            return False

    def get_alternative_search_terms(self, category):
        """
        Генерирует альтернативные поисковые запросы на основе категории.

        Args:
            category (str): Исходная категория.

        Returns:
            list: Список альтернативных поисковых запросов.
        """
        # Базовый словарь альтернатив для разных категорий
        alternatives = {
            "Пэстис эротик": ["Пэстисы", "Наклейки на грудь", "Украшения на грудь", "Ниппель тэйп", "Пестис"],
            "Вибраторы": ["Вибратор для женщин", "Женский вибратор", "Вибратор интимный", "Клиторальный вибратор"],
            "Фаллоимитаторы": ["Дилдо", "Фаллоимитатор для женщин", "Фаллос", "Реалистичный фаллоимитатор"],
            "Анальные пробки": ["Анальная пробка", "Анальный стимулятор", "Анальная игрушка", "Анальный плаг"],
            "Мастурбаторы мужские": ["Мастурбатор", "Мужской мастурбатор", "Искусственная вагина", "Fleshlight"],
            "Страпоны": ["Страпон", "Страпон для пары", "Женский страпон", "Страпон с вибрацией"],
            "Пульсаторы": ["Пульсатор секс", "Пульсирующий вибратор", "Секс пульсатор", "Thrusting вибратор"],
            "БДСМ комплекты": ["БДСМ набор", "Набор БДСМ", "БДСМ комплект", "БДСМ аксессуары набор"],
            "Презервативы": ["Контрацептивы", "Презервативы ультратонкие", "Durex", "Contex"],
            "Лубриканты": ["Смазка интимная", "Гель-смазка", "Лубрикант", "Интимная смазка"],
            "Зажимы для сосков": ["Зажимы на соски", "Прищепки для сосков", "Зажимы на грудь", "Nipple clamps"],
            "Анальные шарики": ["Анальные бусы", "Анальная цепочка", "Анальные шары", "Anal beads"],
            "Анальные бусы": ["Анальные шарики", "Анальная цепочка с шариками", "Гирлянда анальная", "Бусы для анального секса"]
        }

        # Если для категории есть предопределенные альтернативы, используем их
        if category in alternatives:
            return [category] + alternatives[category]

        # Генерируем стандартные вариации
        result = [category]  # Исходная категория всегда первая

        # Удаляем "эротик" и другие суффиксы
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

        # Для категорий, содержащих несколько слов
        words = category.split()
        if len(words) > 1:
            # Основное слово отдельно
            main_word = words[0]  # или другая логика определения основного слова
            if len(main_word) > 3:  # Проверка, чтобы не добавлять предлоги
                result.append(main_word)

            # Перестановки слов
            for i in range(1, len(words)):
                rotated = ' '.join(words[i:] + words[:i])
                if rotated != category:
                    result.append(rotated)

        # Для эротических товаров добавляем вариации
        adult_keywords = ["интимный", "секс", "для взрослых"]
        if any(word in category.lower() for word in ["эротик", "секс", "бдсм", "анальн"]):
            for word in adult_keywords:
                if word not in category.lower():
                    result.append(f"{category} {word}")

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
        Проверяет, можно ли переключиться на следующий поисковый термин.

        Args:
            alternative_terms (list): Список альтернативных поисковых запросов.
            current_idx (int): Текущий индекс поискового запроса.

        Returns:
            bool: True, если можно переключиться на следующий запрос, иначе False.
        """
        return current_idx < len(alternative_terms) - 1

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

    def collect_product_urls(self, category, target_count):
        """
        Собирает URL-адреса товаров для указанной категории.

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
        error_count = 0

        # Alternative search terms
        alternative_search_terms = self.get_alternative_search_terms(category)
        current_search_idx = 0
        current_search_term = category

        while len(urls) < target_count:
            # Check circuit breaker
            if self.circuit_breaker["is_open"]:
                current_time = time.time()
                if self.circuit_open_time and (current_time - self.circuit_open_time > self.circuit_breaker["reset_after"]):
                    # Reset circuit breaker
                    self.circuit_breaker["is_open"] = False
                    self.circuit_breaker["failures"] = 0
                    logger.info("Circuit breaker reset after cooling period")
                else:
                    logger.warning("Circuit breaker open, pausing requests")
                    time.sleep(30)  # Wait before checking again
                    continue

            # Adaptive delay based on failure rate
            if consecutive_failures > 0:
                delay = base_delay * (1.5 ** consecutive_failures)
                logger.info(f"Increasing delay to {delay:.2f}s due to failures")
                time.sleep(delay)

            # Check page limit for current search term
            if page > MAX_PAGES_PER_TERM:
                logger.warning(f"Достигнут предел страниц ({MAX_PAGES_PER_TERM}) для '{current_search_term}'")
                if current_search_idx < len(alternative_search_terms) - 1:
                    current_search_idx += 1
                    current_search_term = alternative_search_terms[current_search_idx]
                    page = 1
                    consecutive_failures = 0
                    logger.info(f"Переключаемся на альтернативный поисковый запрос: '{current_search_term}'")
                    continue
                else:
                    logger.warning(f"Исчерпаны все поисковые запросы. Завершаем с {len(urls)} товарами.")
                    break

            try:
                # Create new driver for each page
                driver = self.browser_manager.get_driver()

                try:
                    search_url = f"https://www.wildberries.ru/catalog/0/search.aspx?search={current_search_term}&page={page}"
                    logger.info(f"Загрузка страницы {page} для поискового запроса '{current_search_term}'")

                    driver.get(search_url)
                    time.sleep(random.uniform(1, 3))  # Короткая пауза после первоначального запроса

                    # Проверяем наличие CAPTCHA
                    if self.captcha_solver and self.captcha_solver.detect_captcha(driver):
                        logger.warning(f"Обнаружена CAPTCHA на странице поиска")
                        if not self.captcha_solver.handle_captcha(driver):
                            logger.error("Не удалось обработать CAPTCHA, переключаемся на другой запрос")
                            consecutive_failures += 1
                            driver.quit()

                            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                                if self.try_switch_to_next_search_term(alternative_search_terms, current_search_idx):
                                    current_search_idx += 1
                                    current_search_term = alternative_search_terms[current_search_idx]
                                    page = 1
                                    consecutive_failures = 0
                                    logger.info(f"Переключаемся на альтернативный поисковый запрос: '{current_search_term}'")
                                else:
                                    logger.error(f"Исчерпаны все альтернативные поисковые запросы. Завершаем сбор ссылок.")
                                    break
                            continue

                    # Проверяем подтверждение возраста, если необходимо
                    self.browser_manager.handle_age_verification(driver)

                    # Ждем загрузки товаров
                    if not self.wait_for_products_load(driver):
                        consecutive_failures += 1
                        logger.warning(
                            f"Не удалось загрузить товары на странице {page} (попытка {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")

                        # Проверяем на CAPTCHA
                        if self.captcha_solver and self.captcha_solver.detect_captcha(driver):
                            logger.warning("Обнаружена CAPTCHA на странице поиска товаров")
                            if not self.captcha_solver.handle_captcha(driver):
                                logger.error("Не удалось обработать CAPTCHA, переключаемся на другой запрос")
                                driver.quit()

                                if self.try_switch_to_next_search_term(alternative_search_terms, current_search_idx):
                                    current_search_idx += 1
                                    current_search_term = alternative_search_terms[current_search_idx]
                                    page = 1
                                    consecutive_failures = 0
                                    logger.info(f"Переключаемся на альтернативный поисковый запрос: '{current_search_term}'")
                                    continue

                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            if self.try_switch_to_next_search_term(alternative_search_terms, current_search_idx):
                                current_search_idx += 1
                                current_search_term = alternative_search_terms[current_search_idx]
                                page = 1
                                consecutive_failures = 0
                                logger.info(f"Переключаемся на альтернативный поисковый запрос: '{current_search_term}'")
                            else:
                                logger.error(f"Исчерпаны все альтернативные поисковые запросы. Завершаем сбор ссылок.")
                                break
                        driver.quit()
                        continue

                    consecutive_failures = 0
                    success = True

                    # Плавная прокрутка
                    self.browser_manager.smooth_scroll(driver,
                                                       scroll_pause_time=random.uniform(0.7, 1.5),
                                                       scroll_increment=random.randint(40, 60))

                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    product_links = soup.select("a.product-card__link.j-card-link.j-open-full-product-card")

                    if not product_links:
                        logger.warning(f"Не найдены ссылки на товары на странице {page}. Пробуем другой селектор.")
                        product_links = soup.select("a.product-card__main.j-card-link")

                        if not product_links:
                            product_links = soup.select("a[href*='/catalog/'][href*='/detail.aspx']")

                            if not product_links:
                                logger.warning(f"Не найдены ссылки на товары через все селекторы на странице {page}")
                                consecutive_failures += 1

                                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                                    logger.warning(
                                        f"Достигнуто максимальное число неудачных попыток подряд. Пробуем другой запрос.")
                                    if self.try_switch_to_next_search_term(alternative_search_terms,
                                                                           current_search_idx):
                                        current_search_idx += 1
                                        current_search_term = alternative_search_terms[current_search_idx]
                                        page = 1
                                        consecutive_failures = 0
                                        logger.info(
                                            f"Переключаемся на альтернативный поисковый запрос: '{current_search_term}'")
                                        driver.quit()
                                        continue
                                    else:
                                        logger.error(
                                            f"Исчерпаны все альтернативные поисковые запросы. Завершаем сбор ссылок.")
                                        break

                                retry_count += 1
                                if retry_count >= max_retries:
                                    logger.error(f"Достигнут лимит попыток для страницы {page}. Переходим к следующей.")
                                    page += 1
                                    retry_count = 0
                                driver.quit()
                                continue

                    retry_count = 0

                    new_urls_count = 0
                    for link in product_links:
                        href = link.get("href")
                        if href:
                            if not href.startswith('http'):
                                href = 'https://www.wildberries.ru' + href

                            article_match = re.search(r'/catalog/(\d+)/detail\.aspx', href)
                            article = article_match.group(1) if article_match else None

                            if article:
                                if article not in self.existing_articles and article not in collected_articles_for_category:
                                    urls.append(href)
                                    collected_articles_for_category.add(article)
                                    new_urls_count += 1
                                    if len(urls) >= target_count:
                                        break
                                else:
                                    logger.debug(f"Артикул {article} уже собран или в списке существующих. Пропускаем.")
                            else:
                                logger.warning(f"Не удалось извлечь артикул из URL: {href}.")
                                if href not in urls:
                                    urls.append(href)
                                    new_urls_count += 1
                                    if len(urls) >= target_count:
                                        break

                    logger.info(
                        f"Страница {page}: добавлено {new_urls_count} новых ссылок (всего: {len(urls)}/{target_count})")

                    if new_urls_count == 0:
                        consecutive_failures += 1

                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            if self.try_switch_to_next_search_term(alternative_search_terms, current_search_idx):
                                current_search_idx += 1
                                current_search_term = alternative_search_terms[current_search_idx]
                                page = 1
                                consecutive_failures = 0
                                logger.info(f"Страницы без новых товаров. Переключаемся на '{current_search_term}'")
                            else:
                                logger.warning(
                                    f"Исчерпаны все альтернативные поисковые запросы. Завершаем с {len(urls)} товарами.")
                                break
                    else:
                        consecutive_failures = 0
                        success = True

                    # Circuit breaker update
                    if not success:
                        self.circuit_breaker["failures"] += 1
                        if self.circuit_breaker["failures"] >= self.circuit_breaker["threshold"]:
                            self.circuit_breaker["is_open"] = True
                            self.circuit_open_time = time.time()
                            logger.warning("Circuit breaker opened due to repeated failures")

                            # Switch search term when circuit breaker opens
                            if current_search_idx < len(alternative_search_terms) - 1:
                                current_search_idx += 1
                                current_search_term = alternative_search_terms[current_search_idx]
                    else:
                        # Reset failures on success
                        self.circuit_breaker["failures"] = max(0, self.circuit_breaker["failures"] - 1)

                    page += 1

                    # Human-like random page skipping
                    if random.random() < 0.2:  # 20% chance to skip pages
                        skip_pages = random.randint(1, 3)
                        page += skip_pages
                        logger.info(f"Randomly skipping {skip_pages} pages to appear more human-like")

                    time.sleep(random.uniform(2, 5))  # Random delay between pages

                finally:
                    driver.quit()
                    logger.debug("Драйвер закрыт после обработки страницы.")

            except Exception as e:
                logger.error(f"Ошибка при сборе ссылок на странице {page}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                consecutive_failures += 1

                # Check if exception looks like a CAPTCHA or bot detection issue
                if any(term in str(e).lower() for term in ["captcha", "challenge", "robot", "автоматизированными"]):
                    logger.warning("Detected possible anti-bot challenge in exception. Cooling down...")
                    time.sleep(random.uniform(60, 120))  # Longer cooldown for suspected blocking

                    # Try switching search terms on suspected blocking
                    if current_search_idx < len(alternative_search_terms) - 1:
                        current_search_idx += 1
                        current_search_term = alternative_search_terms[current_search_idx]
                        page = 1
                        consecutive_failures = 0
                        logger.info(f"Suspected blocking, switching to: '{current_search_term}'")
                else:
                    time.sleep(5)  # Regular pause after other errors

        return urls[:target_count]