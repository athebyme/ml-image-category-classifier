"""
Модуль для работы с веб-браузером в краулере.
"""
import os
import time
import random
import traceback
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from fake_useragent import UserAgent
from ..logging_setup import logger
from ..config import USER_AGENTS, SCREEN_RESOLUTIONS, LANGUAGES, COOKIES_DIR


class BrowserManager:
    """
    Класс для управления веб-браузером.

    Создает, настраивает и управляет экземплярами браузера для краулинга.
    """

    def __init__(self, proxy_manager=None):
        """
        Инициализирует менеджер браузера.

        Args:
            proxy_manager (ProxyManager, optional): Менеджер прокси для использования с браузером.
        """
        self.proxy_manager = proxy_manager
        self.active_drivers = []

    def get_driver(self, use_proxy=False, use_undetected=True):
        """
        Создает и настраивает экземпляр веб-драйвера.

        Args:
            use_proxy (bool): Использовать ли прокси.
            use_undetected (bool): Использовать ли undetected_chromedriver для обхода обнаружения.

        Returns:
            WebDriver: Настроенный экземпляр веб-драйвера.
        """
        if use_undetected:
            return self._get_undetected_driver(use_proxy)
        else:
            return self._get_standard_driver(use_proxy)

    def _get_undetected_driver(self, use_proxy=False):
        """
        Creates an undetected_chromedriver instance to bypass detection.

        Args:
            use_proxy (bool): Whether to use a proxy.

        Returns:
            WebDriver: An undetected_chromedriver instance.
        """
        try:
            # First check if setuptools is installed (required for distutils)
            try:
                import setuptools
            except ImportError:
                logger.warning("setuptools not installed, which is required for undetected_chromedriver")
                logger.warning("Install setuptools: pip install setuptools")
                return self._get_standard_driver(use_proxy)

            # Now try to import undetected_chromedriver
            try:
                import undetected_chromedriver as uc
            except ImportError as e:
                if "No module named 'distutils'" in str(e):
                    logger.error("Missing distutils module which is required by undetected_chromedriver")
                    logger.warning("Install setuptools: pip install setuptools")
                else:
                    logger.error(f"Error importing undetected_chromedriver: {e}")

                logger.warning("Falling back to standard Chrome WebDriver")
                return self._get_standard_driver(use_proxy)

            # Setup undetected-chromedriver options
            options = uc.ChromeOptions()
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")

            # Randomize window size
            width, height = random.choice(SCREEN_RESOLUTIONS)
            options.add_argument(f"--window-size={width},{height}")

            # Randomize language
            options.add_argument(f"--lang={random.choice(LANGUAGES)}")

            # Add proxy if needed
            if use_proxy and self.proxy_manager:
                proxy = self.proxy_manager.get_next_proxy()
                if proxy:
                    options.add_argument(f'--proxy-server={proxy}')

            # Create undetected-chromedriver instance
            driver = uc.Chrome(options=options)

            # Set timeouts
            driver.set_page_load_timeout(30)
            driver.set_script_timeout(30)

            # Test if driver works
            driver.get("about:blank")

            logger.info("Successfully created undetected-chromedriver instance")
            self.active_drivers.append(driver)
            return driver

        except Exception as e:
            logger.error(f"Error creating undetected-chromedriver: {e}")
            logger.error(traceback.format_exc())

            # If undetected-chromedriver fails, use standard Chrome
            time.sleep(3)
            return self._get_standard_driver(use_proxy)

    def _get_standard_driver(self, use_proxy=False):
        """
        Создает стандартный экземпляр веб-драйвера Chrome.

        Args:
            use_proxy (bool): Использовать ли прокси.

        Returns:
            WebDriver: Экземпляр Chrome WebDriver.
        """
        try:
            # Опции Chrome
            options = webdriver.ChromeOptions()
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--disable-extensions")

            # Меры против обнаружения автоматизации
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)

            # Рандомизация размера окна
            width, height = random.choice(SCREEN_RESOLUTIONS)
            options.add_argument(f"--window-size={width},{height}")

            # Добавляем User-Agent
            try:
                ua = UserAgent()
                user_agent = ua.random
            except:
                user_agent = random.choice(USER_AGENTS)

            options.add_argument(f"user-agent={user_agent}")

            # Добавляем прокси при необходимости
            if use_proxy and self.proxy_manager:
                proxy = self.proxy_manager.get_next_proxy()
                if proxy:
                    options.add_argument(f'--proxy-server={proxy}')

            # Создаем веб-драйвер
            driver = webdriver.Chrome(options=options)

            # Добавляем stealth JS для маскировки автоматизации
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                window.chrome = {
                    runtime: {}
                };
                """
            })

            # Устанавливаем таймауты
            driver.set_page_load_timeout(30)
            driver.set_script_timeout(30)

            # Очищаем куки
            driver.delete_all_cookies()

            # Проверяем, работает ли драйвер
            driver.get("about:blank")

            logger.info("Успешно создан экземпляр стандартного Chrome WebDriver")
            self.active_drivers.append(driver)
            return driver

        except Exception as e:
            logger.error(f"Ошибка при создании стандартного Chrome WebDriver: {e}")
            logger.error(traceback.format_exc())
            raise

    def load_cookies(self, driver, category):
        """
        Загружает cookies из предыдущей сессии, если доступны.

        Args:
            driver (WebDriver): Экземпляр веб-драйвера.
            category (str): Категория товаров для идентификации cookies.

        Returns:
            bool: True, если cookies были загружены успешно, иначе False.
        """
        import json

        cookie_file = os.path.join(COOKIES_DIR, f"{category.replace(' ', '_')}.json")

        if os.path.exists(cookie_file):
            with open(cookie_file, 'r') as f:
                cookies = json.load(f)
                for cookie in cookies:
                    if 'expiry' in cookie:
                        del cookie['expiry']  # Удаляем срок действия, чтобы избежать ошибок
                    try:
                        driver.add_cookie(cookie)
                    except:
                        pass
            return True
        return False

    def save_cookies(self, driver, category):
        """
        Сохраняет cookies для будущего использования.

        Args:
            driver (WebDriver): Экземпляр веб-драйвера.
            category (str): Категория товаров для идентификации cookies.
        """
        import json

        cookie_file = os.path.join(COOKIES_DIR, f"{category.replace(' ', '_')}.json")
        cookies = driver.get_cookies()

        with open(cookie_file, 'w') as f:
            json.dump(cookies, f)

        logger.debug(f"Cookies сохранены для категории '{category}'")

    def simulate_human_behavior(self, driver):
        """
        Симулирует поведение человека для обхода обнаружения бота.

        Args:
            driver (WebDriver): Экземпляр веб-драйвера.
        """
        try:
            # Получаем размеры видимой области
            viewport_width = driver.execute_script("return window.innerWidth")
            viewport_height = driver.execute_script("return window.innerHeight")

            # Создаем цепочку действий
            action = ActionChains(driver)

            # 2-3 случайных движения мыши
            for _ in range(random.randint(2, 3)):
                x = random.randint(10, viewport_width - 10)
                y = random.randint(10, viewport_height - 10)
                action.move_by_offset(x, y)
                time.sleep(random.uniform(0.1, 0.3))

            # Иногда кликаем на пустое пространство
            if random.random() < 0.3:  # 30% вероятность
                action.click()

            action.perform()

            # Иногда прокручиваем страницу немного вниз и обратно
            if random.random() < 0.4:  # 40% вероятность
                scroll_amount = random.randint(100, 300)
                driver.execute_script(f"window.scrollBy(0, {scroll_amount})")
                time.sleep(random.uniform(0.5, 1.2))
                driver.execute_script(f"window.scrollBy(0, {-scroll_amount})")

        except Exception as e:
            logger.debug(f"Ошибка при симуляции поведения человека: {e}")

    def smooth_scroll(self, driver, scroll_pause_time=0.5, scroll_increment=50, max_attempts_without_new=5):
        """
        Плавно прокручивает страницу до конца с паузами для загрузки динамического контента.

        Args:
            driver (WebDriver): Экземпляр веб-драйвера.
            scroll_pause_time (float): Время паузы между прокрутками.
            scroll_increment (int): Инкремент прокрутки в пикселях.
            max_attempts_without_new (int): Максимальное число попыток без появления нового контента.
        """
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
                        f"Новых данных не подгрузилось, попытка {attempts_without_new}/{max_attempts_without_new}"
                    )
                else:
                    attempts_without_new = 0
                    last_height = new_height
                current_position = new_height

        logger.info("Прокрутка страницы завершена.")

    def handle_age_verification(self, driver):
        """
        Обрабатывает диалог проверки возраста.

        Args:
            driver (WebDriver): Экземпляр веб-драйвера.

        Returns:
            bool: True, если диалог был обработан успешно, иначе False.
        """
        try:
            # Проверяем наличие кнопки подтверждения возраста
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            try:
                button_present = WebDriverWait(driver, 3).until(
                    EC.presence_of_element_located((By.XPATH,
                                                    "//button[contains(text(), 'Да, мне есть 18') or contains(text(), 'Да, мне исполнилось 18')]"))
                )

                if button_present:
                    # Пробуем нажать кнопку через JavaScript для надежности
                    try:
                        driver.execute_script("arguments[0].click();", button_present)
                        logger.info("Подтвердили возраст через JavaScript")
                        time.sleep(1)  # Даем время на обработку клика
                        return True
                    except Exception as js_err:
                        logger.warning(f"Ошибка при JavaScript клике на кнопку возраста: {js_err}")

                        # Пробуем обычный клик
                        try:
                            button_present.click()
                            logger.info("Подтвердили возраст через обычный клик")
                            time.sleep(1)
                            return True
                        except Exception as click_err:
                            logger.warning(f"Ошибка при обычном клике на кнопку возраста: {click_err}")

                            # Последняя попытка через Action Chains
                            try:
                                actions = ActionChains(driver)
                                actions.move_to_element(button_present).click().perform()
                                logger.info("Подтвердили возраст через ActionChains")
                                time.sleep(1)
                                return True
                            except Exception as action_err:
                                logger.warning(f"Ошибка при использовании ActionChains: {action_err}")
                                return False
            except:
                # Диалог проверки возраста не найден или таймаут
                return False

        except Exception as e:
            logger.debug(f"Ошибка при попытке проверки на страницу возраста: {e}")
            return False

    def close_all_drivers(self):
        """
        Закрывает все активные экземпляры драйвера.
        """
        logger.info(f"Закрытие {len(self.active_drivers)} активных драйверов...")
        for driver in self.active_drivers[:]:
            try:
                driver.quit()
                self.active_drivers.remove(driver)
            except Exception as e:
                logger.error(f"Ошибка при закрытии драйвера: {e}")

        # Для надежности убиваем все процессы Chrome
        try:
            import subprocess
            subprocess.run(['pkill', '-f', 'chrome'], stderr=subprocess.DEVNULL)
        except Exception as e:
            logger.error(f"Ошибка при завершении процессов Chrome: {e}")