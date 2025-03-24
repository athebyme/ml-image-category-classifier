"""
Главный модуль краулера Wildberries.
"""
import time
import random
import signal
import math
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty

from .logging_setup import logger, setup_virtual_display
from .utils.browser import BrowserManager
from .utils.proxy_manager import ProxyManager
from .utils.captcha_resolver import CaptchaSolver
from .utils.backoff import ExponentialBackoff
from .parsers.wildberries_product_parser import WildberriesProductParser
from .parsers.wildberries_search_parser import WildberriesSearchParser
from .storage.json_storage import JsonStorage
from .config import DEFAULT_MAX_WORKERS


class WildberriesCrawler:
    """
    Основной класс краулера для сбора данных с сайта Wildberries.

    Координирует все процессы краулинга: поиск товаров по категориям,
    парсинг страниц товаров, обработку ошибок и сохранение данных.
    """

    def __init__(self, category_targets, max_workers=DEFAULT_MAX_WORKERS, output_dir=None):
        """
        Инициализирует краулер Wildberries.

        Args:
            category_targets (dict): Словарь с целевыми категориями и количеством товаров.
            max_workers (int, optional): Максимальное количество рабочих потоков.
            output_dir (str, optional): Директория для сохранения результатов.
        """
        self.category_targets = category_targets
        self.max_workers = max_workers

        # Настройка виртуального дисплея
        self.display = setup_virtual_display()

        # Инициализация компонентов
        self.proxy_manager = ProxyManager()
        self.browser_manager = BrowserManager(self.proxy_manager)
        self.captcha_solver = CaptchaSolver()
        self.product_parser = WildberriesProductParser(self.browser_manager, self.captcha_solver)
        self.storage = JsonStorage(output_dir)
        self.search_parser = WildberriesSearchParser(self.browser_manager, self.captcha_solver)

        # Настройка хранилища и очереди
        self.search_parser.set_existing_articles(self.storage.get_existing_articles())
        self.products_queue = Queue()
        self.backoff = ExponentialBackoff()

        # Счетчики и статистика
        self.session_count = 0
        self.max_requests_per_session = random.randint(15, 25)
        self.rate_limit_attempts = 0
        self.urls_collected = 0
        self.products_processed = 0
        self.processed_categories = []

        # Настройка обработчиков сигналов для корректного завершения
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

        logger.info("Инициализация WildberriesCrawler завершена.")

    def handle_shutdown(self, signum, frame):
        """
        Обрабатывает корректное завершение при получении сигналов.

        Args:
            signum: Номер сигнала.
            frame: Текущий стековый фрейм.
        """
        signal_names = {
            signal.SIGINT: "SIGINT (Ctrl+C)",
            signal.SIGTERM: "SIGTERM"
        }

        signal_name = signal_names.get(signum, f"сигнал {signum}")
        logger.warning(f"Получен сигнал завершения {signal_name}. Выполняется очистка ресурсов...")

        # Сохраняем текущее состояние
        try:
            # Записываем в файл информацию о состоянии краулера
            status_info = {
                "timestamp": time.time(),
                "categories_processed": self.processed_categories,
                "urls_collected": self.urls_collected,
                "products_processed": self.products_processed,
                "source": "wildberries"  # Добавляем метку источника
            }

            self.storage.save_state(status_info)

        except Exception as e:
            logger.error(f"Ошибка при сохранении состояния: {e}")

        # Закрываем все драйверы
        try:
            logger.info("Закрытие всех драйверов...")
            self.browser_manager.close_all_drivers()
        except Exception as e:
            logger.error(f"Ошибка при закрытии драйверов: {e}")

        # Останавливаем виртуальный дисплей
        try:
            if hasattr(self, 'display') and self.display:
                logger.info("Остановка виртуального дисплея...")
                self.display.stop()
        except Exception as e:
            logger.error(f"Ошибка при остановке виртуального дисплея: {e}")

        # Убиваем все процессы Chrome
        try:
            logger.info("Завершение процессов Chrome...")
            import subprocess
            subprocess.run(['pkill', '-f', 'chrome'], stderr=subprocess.DEVNULL)
            time.sleep(1)  # Ждем завершения процессов
        except Exception as e:
            logger.error(f"Ошибка при завершении процессов Chrome: {e}")

        # Финальное сообщение
        logger.info("Завершение работы краулера Wildberries...")

        # Корректно завершаем работу программы
        import sys
        sys.exit(0)

    def worker(self):
        """
        Рабочий процесс для обработки задач из очереди.

        Создает экземпляр веб-драйвера, обрабатывает URL-адреса товаров из очереди
        и сохраняет данные.
        """
        driver = self.browser_manager.get_driver()
        try:
            while True:
                try:
                    url, category = self.products_queue.get_nowait()
                    try:
                        product_data = self.product_parser.process_product_page(driver, url, category)
                        if product_data:
                            self.storage.save_product(product_data)
                            self.products_processed += 1
                    except Exception as e:
                        # Улучшенное логирование ошибок с трассировкой
                        logger.error(f"Ошибка при обработке URL {url}: {e}")
                        logger.error(traceback.format_exc())
                    finally:
                        self.products_queue.task_done()
                except Empty:
                    break
                except Exception as e:
                    logger.error(f"Неожиданная ошибка в воркере: {e}")
        finally:
            driver.quit()
            logger.debug("Драйвер закрыт.")

    def handle_rate_limit(self):
        """
        Применяет экспоненциальную задержку при обнаружении ограничения запросов.

        Returns:
            float: Фактическое время задержки в секундах.
        """
        # Базовая задержка - начинаем с 5 секунд
        base_delay = 5

        # Увеличиваем задержку с каждой попыткой, но не более 5 минут
        attempt = getattr(self, 'rate_limit_attempts', 0) + 1
        setattr(self, 'rate_limit_attempts', attempt)

        # Формула для экспоненциальной задержки: базовая_задержка * 2^попытка
        delay = min(base_delay * (2 ** attempt), 300)  # Максимум 300 секунд (5 минут)

        # Добавляем случайное отклонение ±20% для маскировки автоматизации
        jitter = random.uniform(0.8, 1.2)
        actual_delay = delay * jitter

        logger.warning(
            f"Обнаружено возможное ограничение запросов. Ожидание {actual_delay:.2f} секунд (попытка {attempt})")

        # Пауза с обратным отсчетом, чтобы видеть прогресс в логах
        for i in range(int(actual_delay), 0, -10):
            remaining = min(i, 10)  # Выводим каждые 10 секунд или меньше
            if i <= 30 or i % 30 == 0:  # Сокращаем количество записей в лог
                logger.debug(f"Осталось ждать: {i} секунд")
            time.sleep(remaining)

        # Сбрасываем счетчик попыток после длительной паузы
        if actual_delay > 60:
            setattr(self, 'rate_limit_attempts', 0)
            logger.info("Счетчик попыток сброшен после длительной паузы")

        return actual_delay

    def run(self):
        """
        Запускает процесс краулинга.

        Собирает URL по категориям, обрабатывает товары и сохраняет данные.
        """
        logger.info("Запуск краулера Wildberries...")

        # Проверяем наличие прокси и при необходимости ищем бесплатные
        if not self.proxy_manager.proxies:
            logger.info("Прокси не найдены в proxies.txt. Поиск бесплатных прокси...")
            proxies = self.proxy_manager.find_free_proxies(count=20)

            if proxies:
                # Тестируем найденные прокси
                self.proxy_manager.test_proxies()

        # Проверяем, активен ли виртуальный дисплей
        if not self.display.is_alive():
            self.display.start()

        start_time = time.time()
        all_tasks = []

        # Определяем оптимальное количество рабочих потоков
        actual_workers = min(4, self.max_workers)

        # Собираем ссылки для всех категорий
        for category, target_count in self.category_targets.items():
            logger.info(f"Сбор ссылок для категории: {category} (цель: {target_count} товаров)")
            product_urls = self.search_parser.collect_product_urls(category, target_count)
            logger.info(f"Для категории {category} собрано {len(product_urls)} ссылок.")

            for url in product_urls:
                all_tasks.append((url, category))

            self.urls_collected += len(product_urls)
            self.processed_categories.append(category)

        # Обрабатываем товары партиями для лучшего управления ресурсами
        logger.info(f"Начало обработки товаров с использованием {actual_workers} рабочих потоков.")
        batch_size = 100
        for i in range(0, len(all_tasks), batch_size):
            batch = all_tasks[i:i + batch_size]
            logger.info(f"Обработка пакета {i // batch_size + 1}/{math.ceil(len(all_tasks) / batch_size)}")

            for url, category in batch:
                self.products_queue.put((url, category))

            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                workers = [executor.submit(self.worker) for _ in range(min(actual_workers, self.products_queue.qsize()))]
                for worker in as_completed(workers):
                    try:
                        worker.result()
                    except Exception as e:
                        logger.error(f"Рабочий поток завершился с ошибкой: {e}")

            logger.info(f"Пакет {i // batch_size + 1} завершен")

            # Случайная пауза между пакетами для уменьшения нагрузки на сервер
            wait_time = random.uniform(30, 60)
            time.sleep(wait_time)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Краулинг Wildberries завершён за {elapsed_time:.2f} секунд.")
        logger.info(f"Всего обработано товаров: {self.products_processed}")
        logger.info(f"Все товары сохранены в каталоге '{self.storage.output_dir}'.")

        # Останавливаем виртуальный дисплей
        try:
            self.display.stop()
        except:
            pass