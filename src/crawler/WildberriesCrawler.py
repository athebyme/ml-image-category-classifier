import logging
import math
import random
import signal
import sys
import time
import os
import json
import base64
import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty

from fake_useragent import UserAgent
from selenium.webdriver import ActionChains, Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
from bs4 import BeautifulSoup
from typing import Dict, List
from loguru import logger
from pyvirtualdisplay import Display

# Настройка логирования: пишем в консоль и в файл

display = Display(visible=0, size=(1920, 1080))
display.start()


logger.remove()
logger.add("crawler.log", format="{time} {level} {message}", level="INFO", rotation="5 MB")


def create_empty_proxies_file(self):
    """
    Создает пустой файл proxies.txt с комментариями о формате
    """
    try:
        with open("proxies.txt", "w", encoding="utf-8") as f:
            f.write("# Файл для списка прокси\n")
            f.write("# Формат: ip:port или user:pass@ip:port\n")
            f.write("# Примеры:\n")
            f.write("# 192.168.1.1:8080\n")
            f.write("# username:password@192.168.1.1:8080\n")
        logger.info("Создан пустой файл proxies.txt")
    except Exception as e:
        logger.error(f"Ошибка при создании файла с прокси: {e}")

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
        self.output_dir = "/home/athebyme/shared_crawler_output"
        self.products_queue = Queue()
        self.session_count = 0
        self.max_requests_per_session = random.randint(15, 25)  # Randomize session length
        self.backoff = ExponentialBackoff()
        os.makedirs(self.output_dir, exist_ok=True)
        self.existing_articles = self.load_existing_articles()
        self.rate_limit_attempts = 0
        self.active_drivers = []  # Для отслеживания активных драйверов
        self.processed_categories = []
        self.urls_collected = 0
        self.products_processed = 0

        # Инициализация атрибутов для прокси
        self.proxy_index = 0
        self.proxies = self.load_proxies()

        # Настройка обработчиков сигналов
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

        self.cookies_dir = "./cookies"
        os.makedirs(self.cookies_dir, exist_ok=True)

        logger.info("Инициализация WildberriesCrawler завершена.")

    def load_cookies(self, driver, category):
        """Load cookies from previous session if available"""
        cookie_file = os.path.join(self.cookies_dir, f"{category.replace(' ', '_')}.json")

        if os.path.exists(cookie_file):
            with open(cookie_file, 'r') as f:
                cookies = json.load(f)
                for cookie in cookies:
                    if 'expiry' in cookie:
                        del cookie['expiry']  # Remove expiry to avoid errors
                    try:
                        driver.add_cookie(cookie)
                    except:
                        pass
            return True
        return False

    def save_cookies(self, driver, category):
        """Save cookies for future use"""
        cookie_file = os.path.join(self.cookies_dir, f"{category.replace(' ', '_')}.json")
        cookies = driver.get_cookies()
        with open(cookie_file, 'w') as f:
            json.dump(cookies, f)

    def detect_captcha(self, driver):
        """
        Detects if we're on a CAPTCHA page
        Returns: True if CAPTCHA is detected, False otherwise
        """
        try:
            # Check for common CAPTCHA identifiers
            captcha_selectors = [
                (By.ID, "__wbaas_captcha_container"),
                (By.XPATH, "//div[contains(@class, 'captcha')]"),
                (By.XPATH, "//title[contains(text(), 'Почти готово')]"),
                (By.XPATH, "//p[contains(text(), 'IP-адрес')]"),
            ]

            for selector_type, selector in captcha_selectors:
                try:
                    element = driver.find_element(selector_type, selector)
                    if element:
                        logger.warning("CAPTCHA detected on page!")
                        return True
                except:
                    continue

            # Also check for patterns in URL or page source that indicate verification
            if "/__wbaas/challenges/captcha/" in driver.current_url or "/__wbaas/challenges/captcha/" in driver.page_source:
                logger.warning("CAPTCHA challenge URL detected!")
                return True

            return False
        except Exception as e:
            logger.error(f"Error while checking for CAPTCHA: {e}")
            return False

    def handle_captcha(self, driver):
        """
        Attempts to handle CAPTCHA challenges with 2captcha
        Returns: True if successfully handled or no CAPTCHA, False if CAPTCHA couldn't be solved
        """
        if not self.detect_captcha(driver):
            return True  # No CAPTCHA detected

        try:
            # Take screenshot for potential manual intervention
            timestamp = int(time.time())
            screenshot_path = f"captcha_{timestamp}.png"
            driver.save_screenshot(screenshot_path)

            # Save page HTML for analysis
            html_path = f"captcha_{timestamp}.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(driver.page_source)

            logger.info(f"CAPTCHA detected! Screenshot saved to {screenshot_path}, HTML to {html_path}")

            # Try to solve using 2captcha (free trial)
            try:
                # Import 2captcha library
                from twocaptcha import TwoCaptcha

                # Find the type of captcha Wildberries is using
                captcha_type = self.identify_captcha_type(driver)

                if captcha_type == "image":
                    # Solve image captcha
                    return self.solve_image_captcha(driver, screenshot_path)
                elif captcha_type == "recaptcha":
                    # Solve reCAPTCHA
                    return self.solve_recaptcha(driver)
                elif captcha_type == "hcaptcha":
                    # Solve hCaptcha
                    return self.solve_hcaptcha(driver)
                else:
                    logger.warning(f"Unknown CAPTCHA type: {captcha_type}")

            except ImportError:
                logger.warning("2captcha library not installed. Run: pip install 2captcha-python")
            except Exception as e:
                logger.error(f"Error using 2captcha: {e}")

            # If we get here, automated solving failed - try manual intervention
            is_headless = "--headless" in str(driver.capabilities.get("chrome", {}).get("chromedriverArgs", []))
            if not is_headless:
                wait_time = 60  # 60 seconds for manual intervention
                logger.info(f"Waiting {wait_time} seconds for manual CAPTCHA solving...")
                time.sleep(wait_time)

                # Check if we're still on CAPTCHA page
                if not self.detect_captcha(driver):
                    logger.info("CAPTCHA appears to have been solved manually!")
                    return True

            # Last resort: try refreshing and see if we get lucky
            logger.warning("Failed to solve CAPTCHA. Trying to refresh the page...")
            driver.refresh()
            time.sleep(5)

            return not self.detect_captcha(driver)  # Return True if CAPTCHA is gone after refresh

        except Exception as e:
            logger.error(f"Error while trying to handle CAPTCHA: {e}")
            return False

    def identify_captcha_type(self, driver):
        """Identify the type of CAPTCHA present on the page"""
        try:
            # Check for image CAPTCHA (typically used by Wildberries)
            if driver.find_elements(By.XPATH, "//img[contains(@src, 'captcha') or contains(@class, 'captcha')]"):
                return "image"

            # Check for reCAPTCHA
            if driver.find_elements(By.XPATH,
                                    "//div[contains(@class, 'g-recaptcha')]") or "www.google.com/recaptcha" in driver.page_source:
                return "recaptcha"

            # Check for hCaptcha
            if driver.find_elements(By.XPATH,
                                    "//div[contains(@class, 'h-captcha')]") or "hcaptcha.com" in driver.page_source:
                return "hcaptcha"

            # Default to image CAPTCHA for Wildberries (most common)
            return "image"
        except:
            return "image"  # Default to image CAPTCHA if detection fails

    def solve_image_captcha(self, driver, screenshot_path):
        """Solve image CAPTCHA using 2captcha"""
        try:
            from twocaptcha import TwoCaptcha, NetworkException
            import time
            import base64
            import os

            # Get your API key - sign up for free trial at 2captcha.com
            api_key = os.environ.get('CAPTCHA_API_KEY', 'YOUR_2CAPTCHA_API_KEY')

            # Create solver instance
            solver = TwoCaptcha(api_key)

            # Try to find the CAPTCHA image element
            captcha_img = None
            possible_selectors = [
                "//img[contains(@src, 'captcha')]",
                "//div[contains(@class, 'captcha')]//img",
                "//div[@id='__wbaas_captcha_container']//img"
            ]

            for selector in possible_selectors:
                try:
                    elements = driver.find_elements(By.XPATH, selector)
                    if elements:
                        captcha_img = elements[0]
                        break
                except:
                    continue

            if captcha_img:
                # Get image directly from the element
                # First try to get src
                img_src = captcha_img.get_attribute('src')

                if img_src and not img_src.startswith('data:'):
                    # Download the image
                    import requests
                    img_response = requests.get(img_src, timeout=10)
                    if img_response.status_code == 200:
                        with open('current_captcha.png', 'wb') as f:
                            f.write(img_response.content)
                        captcha_file = 'current_captcha.png'
                    else:
                        # If download fails, use screenshot
                        captcha_file = screenshot_path
                else:
                    # Take screenshot of just the CAPTCHA image
                    captcha_img.screenshot('captcha_element.png')
                    captcha_file = 'captcha_element.png'
            else:
                # If we can't find the specific element, use the full screenshot
                captcha_file = screenshot_path

            logger.info(f"Sending CAPTCHA image to 2captcha service...")

            try:
                # Send CAPTCHA for solving
                result = solver.normal(captcha_file)
                captcha_text = result.get('code', '')

                logger.info(f"2captcha solution received: {captcha_text}")

                # Find the input field and submit button
                input_field = None
                try:
                    # Try different potential selectors for the input field
                    for selector in ["input[name='captcha']", "input[id*='captcha']", "input[type='text']"]:
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            input_field = elements[0]
                            break
                except:
                    pass

                # If we found an input field, enter the solution
                if input_field:
                    input_field.clear()
                    input_field.send_keys(captcha_text)

                    # Find and click the submit button
                    submit_button = None
                    try:
                        # Try different potential selectors for the submit button
                        for selector in ["button[type='submit']", "input[type='submit']", "button"]:
                            elements = driver.find_elements(By.CSS_SELECTOR, selector)
                            if elements:
                                submit_button = elements[0]
                                break
                    except:
                        pass

                    if submit_button:
                        submit_button.click()
                        time.sleep(3)  # Wait for submission to process

                        # Check if CAPTCHA is still there
                        return not self.detect_captcha(driver)
                    else:
                        logger.warning("Could not find submit button for CAPTCHA")
                else:
                    logger.warning("Could not find input field for CAPTCHA")

                return False  # CAPTCHA not solved

            except NetworkException as e:
                logger.error(f"Network error with 2captcha: {e}")
                return False
            except Exception as e:
                logger.error(f"Error solving image CAPTCHA: {e}")
                return False
        except Exception as e:
            logger.error(f"General error in solve_image_captcha: {e}")
            return False

    def solve_recaptcha(self, driver):
        """Solve Google reCAPTCHA using 2captcha"""
        try:
            from twocaptcha import TwoCaptcha
            import time

            # Get your API key - sign up for free trial at 2captcha.com
            api_key = os.environ.get('CAPTCHA_API_KEY', 'YOUR_2CAPTCHA_API_KEY')

            # Create solver instance
            solver = TwoCaptcha(api_key)

            # Get site key
            site_key = None
            try:
                # Try to find the site key in the page source
                recaptcha_div = driver.find_element(By.CSS_SELECTOR, ".g-recaptcha")
                site_key = recaptcha_div.get_attribute("data-sitekey")
            except:
                # Try to extract from page source
                import re
                match = re.search(r'data-sitekey=["\']([^"\']+)["\']', driver.page_source)
                if match:
                    site_key = match.group(1)

            if not site_key:
                logger.error("Could not find reCAPTCHA site key")
                return False

            logger.info(f"Found reCAPTCHA site key: {site_key}")

            # Send CAPTCHA for solving
            result = solver.recaptcha(
                sitekey=site_key,
                url=driver.current_url
            )

            g_response = result.get('code')
            logger.info("Received reCAPTCHA solution from 2captcha")

            # Apply the solution
            driver.execute_script(f"document.getElementById('g-recaptcha-response').innerHTML = '{g_response}';")

            # Submit the form - try to find the submit button
            try:
                submit_buttons = driver.find_elements(By.XPATH,
                                                      "//button[@type='submit'] | //input[@type='submit'] | //button[contains(text(), 'Submit')]")
                if submit_buttons:
                    submit_buttons[0].click()
                else:
                    # Try to trigger form submission via JavaScript
                    driver.execute_script("""
                        var forms = document.getElementsByTagName('form');
                        if (forms.length > 0) {
                            forms[0].submit();
                        }
                    """)
            except Exception as e:
                logger.warning(f"Error submitting reCAPTCHA form: {e}")

            time.sleep(3)  # Wait for submission to process

            # Check if CAPTCHA is still there
            return not self.detect_captcha(driver)

        except Exception as e:
            logger.error(f"Error solving reCAPTCHA: {e}")
            return False

    def solve_hcaptcha(self, driver):
        """Solve hCaptcha using 2captcha"""
        try:
            from twocaptcha import TwoCaptcha
            import time

            # Get your API key - sign up for free trial at 2captcha.com
            api_key = os.environ.get('CAPTCHA_API_KEY', '')

            # Create solver instance
            solver = TwoCaptcha(api_key)

            # Get site key
            site_key = None
            try:
                # Try to find the site key in the page source
                hcaptcha_div = driver.find_element(By.CSS_SELECTOR, ".h-captcha")
                site_key = hcaptcha_div.get_attribute("data-sitekey")
            except:
                # Try to extract from page source
                import re
                match = re.search(r'data-sitekey=["\']([^"\']+)["\']', driver.page_source)
                if match:
                    site_key = match.group(1)

            if not site_key:
                logger.error("Could not find hCaptcha site key")
                return False

            logger.info(f"Found hCaptcha site key: {site_key}")

            # Send CAPTCHA for solving
            result = solver.hcaptcha(
                sitekey=site_key,
                url=driver.current_url
            )

            h_response = result.get('code')
            logger.info("Received hCaptcha solution from 2captcha")

            # Apply the solution
            driver.execute_script(f"document.getElementsByName('h-captcha-response')[0].innerHTML = '{h_response}';")

            # Submit the form - try to find the submit button
            try:
                submit_buttons = driver.find_elements(By.XPATH,
                                                      "//button[@type='submit'] | //input[@type='submit'] | //button[contains(text(), 'Submit')]")
                if submit_buttons:
                    submit_buttons[0].click()
                else:
                    # Try to trigger form submission via JavaScript
                    driver.execute_script("""
                        var forms = document.getElementsByTagName('form');
                        if (forms.length > 0) {
                            forms[0].submit();
                        }
                    """)
            except Exception as e:
                logger.warning(f"Error submitting hCaptcha form: {e}")

            time.sleep(3)  # Wait for submission to process

            # Check if CAPTCHA is still there
            return not self.detect_captcha(driver)

        except Exception as e:
            logger.error(f"Error solving hCaptcha: {e}")
            return False

    def process_request(self, driver, url, category, retry_count=0, max_retries=3):
        """
        Complete request pipeline with CAPTCHA and anti-bot protection

        Args:
            driver: WebDriver instance
            url: URL to request
            category: Product category
            retry_count: Current retry attempt
            max_retries: Maximum number of retries

        Returns:
            True if request was successful, False otherwise
        """
        if retry_count >= max_retries:
            logger.error(f"Maximum retries reached for URL: {url}")
            return False

        try:
            # 1. Add randomized delay before request
            delay = random.uniform(2, 8) * (1 + (retry_count * 0.5))  # Increase delay with each retry
            logger.debug(f"Waiting {delay:.2f}s before requesting {url}")
            time.sleep(delay)

            # 2. Make the request
            logger.info(f"Requesting URL: {url}")
            driver.get(url)

            # 3. Handle CAPTCHA if present
            if self.detect_captcha(driver):
                logger.warning(f"CAPTCHA detected on {url}")
                if not self.handle_captcha(driver):
                    # CAPTCHA handling failed, try again with new session
                    logger.warning("CAPTCHA handling failed, resetting session and retrying...")
                    driver.quit()
                    new_driver = self.get_driver()
                    return self.process_request(new_driver, url, category, retry_count + 1, max_retries)

            # 4. Handle age verification if needed
            self.handle_age_verification(driver)

            # 5. Check if the page loaded correctly
            try:
                # Wait for something on the page that indicates successful loading
                # For search pages:
                if "search.aspx" in url:
                    try:
                        WebDriverWait(driver, 20).until(
                            EC.presence_of_element_located(
                                (By.CSS_SELECTOR, ".product-card, a.j-card-link, .catalog-page__content"))
                        )
                    except TimeoutException:
                        logger.warning(f"Timeout waiting for product cards on search page: {url}")
                        # Verify if it's a no-results page or an actual error
                        if "По Вашему запросу ничего не найдено" in driver.page_source:
                            logger.info("Search returned no results")
                            return True  # No results is still a valid response
                        return False

                # For product detail pages:
                elif "detail.aspx" in url:
                    try:
                        WebDriverWait(driver, 20).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, ".product-page__title, h1.name"))
                        )
                    except TimeoutException:
                        logger.warning(f"Timeout waiting for product title on detail page: {url}")
                        return False
            except Exception as e:
                logger.error(f"Error checking page load: {e}")
                return False

            # Add some random scrolling to appear more human-like
            self.simulate_human_behavior(driver)

            return True  # Request successful!

        except Exception as e:
            logger.error(f"Error in request pipeline: {e}")

            # Check if it's a common error that merits retry
            if any(s in str(e) for s in ["timeout", "not reachable", "crashed", "session"]):
                logger.warning(f"Retrying due to connection error ({retry_count + 1}/{max_retries})")
                # Use a new driver for the retry
                try:
                    driver.quit()
                except:
                    pass

                time.sleep(5 * (2 ** retry_count))  # Exponential backoff
                new_driver = self.get_driver()
                return self.process_request(new_driver, url, category, retry_count + 1, max_retries)

            return False

    def handle_rate_limit(self):
        """
        Применяет экспоненциальную задержку при обнаружении ограничения запросов
        без использования прокси
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

    def handle_shutdown(self, signum, frame):
        """
        Обрабатывает корректное завершение при получении сигналов
        """
        signal_names = {
            signal.SIGINT: "SIGINT (Ctrl+C)",
            signal.SIGTERM: "SIGTERM"
        }

        signal_name = signal_names.get(signum, f"сигнал {signum}")
        logger.warning(f"Получен сигнал завершения {signal_name}. Выполняется очистка ресурсов...")

        # Сохраняем текущее состояние, если нужно
        try:
            # Записываем в файл информацию о состоянии краулера
            status_info = {
                "timestamp": time.time(),
                "categories_processed": getattr(self, 'processed_categories', []),
                "urls_collected": getattr(self, 'urls_collected', 0),
                "products_processed": getattr(self, 'products_processed', 0)
            }

            import json
            with open("crawler_state.json", "w", encoding="utf-8") as f:
                json.dump(status_info, f, ensure_ascii=False, indent=2)

            logger.info("Состояние краулера сохранено в crawler_state.json")
        except Exception as e:
            logger.error(f"Ошибка при сохранении состояния: {e}")

        # Закрываем все драйверы
        try:
            logger.info("Закрытие всех драйверов...")
            # Если у вас есть список активных драйверов
            active_drivers = getattr(self, 'active_drivers', [])
            for driver in active_drivers:
                try:
                    driver.quit()
                except:
                    pass
        except Exception as e:
            logger.error(f"Ошибка при закрытии драйверов: {e}")

        # Если используется виртуальный дисплей
        try:
            if 'display' in globals() and display:
                logger.info("Остановка виртуального дисплея...")
                display.stop()
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
        logger.info("Завершение работы краулера...")

        # Корректно завершаем работу программы
        sys.exit(0)

    def get_driver(self):
        import undetected_chromedriver as uc
        import random
        import time

        try:
            # Setup undetected-chromedriver options
            options = uc.ChromeOptions()
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")

            # If headless is needed (note that headless might be detected more easily)
            # options.add_argument("--headless")

            # Randomize window size
            resolutions = [(1366, 768), (1920, 1080), (1440, 900), (1536, 864)]
            width, height = random.choice(resolutions)
            options.add_argument(f"--window-size={width},{height}")

            # Add language settings to vary fingerprint
            languages = ["en-US,en;q=0.9", "en-GB,en;q=0.9", "ru-RU,ru;q=0.9", "de-DE,de;q=0.9"]
            options.add_argument(f"--lang={random.choice(languages)}")

            # Add proxy if available
            if hasattr(self, 'proxies') and self.proxies:
                proxy = random.choice(self.proxies)
                options.add_argument(f'--proxy-server={proxy}')

            # Create the undetected-chromedriver
            driver = uc.Chrome(options=options)

            # Set timeouts (these might be handled differently in undetected_chromedriver)
            driver.set_page_load_timeout(30)
            driver.set_script_timeout(30)

            # Test if driver works
            driver.get("about:blank")

            logger.info("Successfully created undetected-chromedriver instance")
            return driver

        except Exception as e:
            logger.error(f"Error creating undetected-chromedriver: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Fallback to regular Chrome if undetected fails
            time.sleep(3)
            try:
                logger.warning("Falling back to regular ChromeDriver")
                from selenium.webdriver.chrome.service import Service

                # Define Chrome options
                options = webdriver.ChromeOptions()
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument("--disable-extensions")

                # Anti-bot detection measures
                options.add_argument("--disable-blink-features=AutomationControlled")
                options.add_experimental_option("excludeSwitches", ["enable-automation"])
                options.add_experimental_option("useAutomationExtension", False)

                # Randomize window size
                width, height = random.choice(resolutions)
                options.add_argument(f"--window-size={width},{height}")

                # Add user agent
                from fake_useragent import UserAgent
                ua = UserAgent()
                user_agent = ua.random
                options.add_argument(f"user-agent={user_agent}")

                # Add proxy if available
                if hasattr(self, 'proxies') and self.proxies:
                    proxy = random.choice(self.proxies)
                    options.add_argument(f'--proxy-server={proxy}')

                # Try standard chromedriver
                driver = webdriver.Chrome(options=options)

                # Add stealth JS
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

                # Set timeouts
                driver.set_page_load_timeout(30)
                driver.set_script_timeout(30)

                # Clear cookies
                driver.delete_all_cookies()

                # Test if driver works
                driver.get("about:blank")

                return driver
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                raise

    def handle_age_verification(self, driver):
        try:
            # First check if age verification element exists using presence_of_element_located
            try:
                button_present = WebDriverWait(driver, 3).until(
                    EC.presence_of_element_located((By.XPATH,
                                                    "//button[contains(text(), 'Да, мне есть 18') or contains(text(), 'Да, мне исполнилось 18')]"))
                )

                if button_present:
                    # Use JavaScript to click for better reliability
                    try:
                        driver.execute_script("arguments[0].click();", button_present)
                        logger.info("Подтвердили возраст через JavaScript")
                        time.sleep(1)  # Give time for the click to take effect
                        return True
                    except Exception as js_err:
                        logger.warning(f"Ошибка при JavaScript клике на кнопку возраста: {js_err}")

                        # Try regular click as fallback
                        try:
                            button_present.click()
                            logger.info("Подтвердили возраст через обычный клик")
                            time.sleep(1)
                            return True
                        except Exception as click_err:
                            logger.warning(f"Ошибка при обычном клике на кнопку возраста: {click_err}")

                            # Final attempt with Action Chains
                            try:
                                from selenium.webdriver.common.action_chains import ActionChains
                                actions = ActionChains(driver)
                                actions.move_to_element(button_present).click().perform()
                                logger.info("Подтвердили возраст через ActionChains")
                                time.sleep(1)
                                return True
                            except Exception as action_err:
                                logger.warning(f"Ошибка при использовании ActionChains: {action_err}")
                                return False
            except:
                # No age verification found or timeout
                return False

        except Exception as e:
            logger.debug(f"Ошибка при попытке проверки на страницу возраста: {e}")
            return False

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

    def parse_wb_slider_images_high_quality(self, driver, logger=None):
        """
        Enhanced method to parse all high-quality images from Wildberries product page.
        Uses multiple strategies to get the best possible images.

        Args:
            driver: WebDriver instance
            logger: Optional logger for recording errors

        Returns:
            list: List of high-quality image URLs
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        image_urls = set()
        max_attempts = 15
        attempts = 0

        try:
            # First detect what type of image gallery we're dealing with
            # Wildberries has multiple gallery layouts depending on the product

            # Wait for the gallery/slider to load
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((
                        By.CSS_SELECTOR,
                        "ul.swiper-wrapper, div.slider-block, div.sw-slider-product, div.zoom-image-container"
                    ))
                )
                time.sleep(1)  # Additional wait to make sure images are fully loaded
            except TimeoutException:
                logger.warning("Не найдена галерея изображений на странице товара")
                return []

            # First try to find big images directly in the page source
            # This is the most reliable and highest quality method
            page_source = driver.page_source

            # Pattern for high-quality image URLs in the page source
            patterns = [
                r'(https:\/\/images\.wbstatic\.net\/big\/new\/\d+\/\d+[-\w]+\.jpg)',
                r'(https:\/\/images\.wbstatic\.net\/large\/new\/\d+\/\d+[-\w]+\.jpg)',
                r'(https:\/\/images\.wbstatic\.net\/c516x688\/new\/\d+\/\d+[-\w]+\.jpg)',
                r'(https:\/\/[\w-]+\.wbstatic\.net\/big\/new\/\d+\/\d+.*?\.jpg)'
            ]

            for pattern in patterns:
                big_images = re.findall(pattern, page_source)
                for img_url in big_images:
                    # Ensure it's a full URL
                    if img_url.startswith('//'):
                        img_url = 'https:' + img_url
                    image_urls.add(img_url)

            # If we found images through regex, return those
            if image_urls:
                logger.info(f"Найдено {len(image_urls)} изображений высокого качества через анализ HTML")
                return list(image_urls)

            # If regex method failed, try extracting through DOM
            logger.info("Попытка извлечения изображений через DOM элементы")

            # Get all visible slides
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
                # Last resort - look for any image that might be a product image
                logger.warning("Не найдены слайды галереи, пробуем найти любые изображения товара")
                current_strategy = "fallback"

                # Try to find any product images
                img_elements = driver.find_elements(By.CSS_SELECTOR, "img[src*='/catalog/']")

                for img in img_elements:
                    src = img.get_attribute("src")
                    if src and not src.startswith("data:"):
                        # Convert to high resolution URL
                        high_res_url = src.replace("/tm/", "/big/").replace("/c246x328/", "/big/")
                        image_urls.add(high_res_url)

                # If we found any images this way, return them
                if image_urls:
                    return list(image_urls)

            # Process slides based on the detected gallery type
            logger.info(f"Используем стратегию извлечения изображений: {current_strategy}")

            # Function to extract image URL and convert to high resolution
            def extract_and_convert_image_url(element):
                try:
                    img_url = None

                    # Check different attributes that might contain the URL
                    for attr in ["src", "data-src", "data-bx-src", "data-original", "data-src-pb"]:
                        img_url = element.get_attribute(attr)
                        if img_url and not img_url.startswith("data:"):
                            break

                    if not img_url or img_url.startswith("data:"):
                        return None

                    # Normalize URL
                    if img_url.startswith('//'):
                        img_url = 'https:' + img_url

                    # Convert to high resolution
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

            # Process slides based on strategy
            if current_strategy == "standard_gallery" or current_strategy == "new_gallery":
                # For each slide in the gallery
                for slide in slides:
                    try:
                        # Find the image element
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

                    # If we haven't found any images yet, try clicking the slide to reveal more images
                    if not image_urls and attempts < 5:
                        try:
                            driver.execute_script("arguments[0].click();", slide)
                            time.sleep(0.5)

                            # Try to find modal images
                            modal_images = driver.find_elements(By.CSS_SELECTOR, "img.photo-zoom__preview")
                            for img in modal_images:
                                img_url = extract_and_convert_image_url(img)
                                if img_url:
                                    image_urls.add(img_url)

                            # Close modal
                            actions = ActionChains(driver)
                            actions.send_keys(Keys.ESCAPE).perform()
                            time.sleep(0.3)
                        except Exception as e:
                            logger.debug(f"Ошибка при попытке кликнуть на слайд: {e}")

                    attempts += 1

                # If we still have no images, try to find the "next" button and navigate through slides
                if len(image_urls) < 3 and attempts < max_attempts:  # We want at least 3 images if possible
                    try:
                        next_button = WebDriverWait(driver, 2).until(
                            EC.element_to_be_clickable(
                                (By.CSS_SELECTOR, "button.swiper-button-next, div.slider-control--next"))
                        )

                        # Check if button is disabled
                        if 'swiper-button-disabled' in next_button.get_attribute('class'):
                            logger.debug("Кнопка Next заблокирована, все слайды просмотрены.")
                        else:
                            # Click and wait for new slides
                            driver.execute_script("arguments[0].click();", next_button)
                            time.sleep(0.7)

                            # Process newly visible slides (recursive call would be cleaner but let's avoid it)
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
                # Find all image elements
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

            # Use our new process_request method instead of direct driver.get()
            if not self.process_request(driver, url, category):
                logger.error(f"Failed to load product page: {url}")
                return None

            # Continue with existing product extraction code
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

            # Rest of your existing extraction code...
            main_soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Product name
            name_element = main_soup.select_one("h1.product-page__title")
            name = name_element.text.strip() if name_element else ""

            # Brand
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

            images = self.parse_wb_slider_images_high_quality(driver, logger)

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
            # First ensure the page itself is fully loaded
            WebDriverWait(driver, 15).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )

            # Add random delays to appear more human-like
            time.sleep(random.uniform(2, 5))

            # Try multiple selectors in sequence with proper error handling
            selectors = [
                ".product-card__wrapper",
                ".product-card",
                "a.j-card-link",
                "[data-card-index]",  # Add more alternative selectors
                ".catalog-page .product-card"
            ]

            for selector in selectors:
                try:
                    WebDriverWait(driver, 8).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
                    )
                    logger.info(f"Products loaded successfully using selector: {selector}")
                    return True
                except TimeoutException:
                    logger.debug(f"Selector {selector} failed, trying next")
                    continue

            # If we got here, all selectors failed
            logger.error("All product selectors failed")

            # Take a screenshot for debugging
            timestamp = int(time.time())
            screenshot_path = f"error_page_{timestamp}.png"
            driver.save_screenshot(screenshot_path)
            logger.info(f"Saved error screenshot: {screenshot_path}")

            return False
        except Exception as e:
            logger.error(f"Unexpected error waiting for products: {e}")
            return False

    def collect_product_urls(self, category: str, target_count: int) -> List[str]:
        urls = []
        page = 1
        max_retries = 3
        retry_count = 0
        max_consecutive_failures = 3
        consecutive_failures = 0
        max_pages_per_term = 50
        collected_articles_for_category = set()
        base_delay = 5
        error_count = 0

        # Circuit breaker pattern to prevent hammering the site after multiple failures
        circuit_breaker = {
            "failures": 0,
            "threshold": 5,
            "is_open": False,
            "reset_after": 180  # seconds
        }

        circuit_open_time = None
        success = False

        # Alternative search terms
        alternative_search_terms = self.get_alternative_search_terms(category)
        current_search_idx = 0
        current_search_term = category

        while len(urls) < target_count:
            # Check circuit breaker
            if circuit_breaker["is_open"]:
                current_time = time.time()
                if circuit_open_time and (current_time - circuit_open_time > circuit_breaker["reset_after"]):
                    # Reset circuit breaker
                    circuit_breaker["is_open"] = False
                    circuit_breaker["failures"] = 0
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
            if page > max_pages_per_term:
                logger.warning(f"Достигнут предел страниц ({max_pages_per_term}) для '{current_search_term}'")
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
                driver = self.get_driver()

                try:
                    search_url = f"https://www.wildberries.ru/catalog/0/search.aspx?search={current_search_term}&page={page}"
                    logger.info(f"Загрузка страницы {page} для поискового запроса '{current_search_term}'")

                    # Use process_request instead of direct driver.get
                    request_successful = self.process_request(driver, search_url, category)

                    if not request_successful:
                        consecutive_failures += 1
                        logger.warning(
                            f"Не удалось загрузить страницу {page} (попытка {consecutive_failures}/{max_consecutive_failures})")
                        if consecutive_failures >= max_consecutive_failures:
                            if self.try_switch_to_next_search_term(alternative_search_terms, current_search_idx):
                                current_search_idx += 1
                                current_search_term = alternative_search_terms[current_search_idx]
                                page = 1
                                consecutive_failures = 0
                                logger.info(
                                    f"Переключаемся на альтернативный поисковый запрос: '{current_search_term}'")
                            else:
                                logger.error(f"Исчерпаны все альтернативные поисковые запросы. Завершаем сбор ссылок.")
                                break
                        continue

                    # No need for additional age verification or human behavior simulation
                    # as process_request handles that

                    if not self.wait_for_products_load(driver):
                        consecutive_failures += 1
                        logger.warning(
                            f"Не удалось загрузить товары на странице {page} (попытка {consecutive_failures}/{max_consecutive_failures})")

                        # Check if this is a CAPTCHA page
                        if self.detect_captcha(driver):
                            logger.warning("Обнаружена CAPTCHA на странице поиска товаров")
                            if not self.handle_captcha(driver):
                                logger.error("Не удалось обработать CAPTCHA, переключаемся на другой запрос")
                                if self.try_switch_to_next_search_term(alternative_search_terms, current_search_idx):
                                    current_search_idx += 1
                                    current_search_term = alternative_search_terms[current_search_idx]
                                    page = 1
                                    consecutive_failures = 0
                                    logger.info(
                                        f"Переключаемся на альтернативный поисковый запрос: '{current_search_term}'")
                                    continue

                        if consecutive_failures >= max_consecutive_failures:
                            if self.try_switch_to_next_search_term(alternative_search_terms, current_search_idx):
                                current_search_idx += 1
                                current_search_term = alternative_search_terms[current_search_idx]
                                page = 1
                                consecutive_failures = 0
                                logger.info(
                                    f"Переключаемся на альтернативный поисковый запрос: '{current_search_term}'")
                            else:
                                logger.error(f"Исчерпаны все альтернативные поисковые запросы. Завершаем сбор ссылок.")
                                break
                        continue

                    consecutive_failures = 0
                    success = True

                    # Human-like scrolling
                    self.smooth_scroll(driver, scroll_pause_time=random.uniform(0.7, 1.5),
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

                                if consecutive_failures >= max_consecutive_failures:
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

                        if consecutive_failures >= max_consecutive_failures:
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
                        circuit_breaker["failures"] += 1
                        if circuit_breaker["failures"] >= circuit_breaker["threshold"]:
                            circuit_breaker["is_open"] = True
                            circuit_open_time = time.time()
                            logger.warning("Circuit breaker opened due to repeated failures")

                            # Switch search term when circuit breaker opens
                            if current_search_idx < len(alternative_search_terms) - 1:
                                current_search_idx += 1
                                current_search_term = alternative_search_terms[current_search_idx]
                    else:
                        # Reset failures on success
                        circuit_breaker["failures"] = max(0, circuit_breaker["failures"] - 1)

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



    def try_switch_to_next_search_term(self, alternative_terms: List[str], current_idx: int) -> bool:
        """Проверяет, можно ли переключиться на следующий поисковый термин"""
        return current_idx < len(alternative_terms) - 1

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

        if not self.proxies:
            logger.info("No proxies found in proxies.txt. Searching for free proxies...")
            self.proxies = self.find_free_proxies(count=20)

            if self.proxies:
                # Test the found proxies
                self.test_proxies()

        if not display.is_alive():
            display.start()

        start_time = time.time()
        all_tasks = []
        actual_workers = min(4, self.max_workers)

        for category, target_count in self.category_targets.items():
            logger.info(f"Сбор ссылок для категории: {category} (цель: {target_count} товаров)")
            product_urls = self.collect_product_urls(category, target_count)
            logger.info(f"Для категории {category} собрано {len(product_urls)} ссылок.")

            for url in product_urls:
                all_tasks.append((url, category))

        logger.info(f"Начало обработки товаров категории: {category} с использованием {self.max_workers} воркеров.")
        batch_size = 100
        for i in range(0, len(all_tasks), batch_size):
            batch = all_tasks[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{math.ceil(len(all_tasks) / batch_size)}")

            for url, category in batch:
                self.products_queue.put((url, category))

            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                workers = [executor.submit(self.worker) for _ in
                           range(min(self.max_workers, self.products_queue.qsize()))]
                for worker in as_completed(workers):
                    try:
                        worker.result()
                    except Exception as e:
                        logger.error(f"Worker failed: {e}")

            logger.info(f"Batch {i // batch_size + 1} complete")

            wait_time = random.uniform(30, 60)
            time.sleep(wait_time)
            logger.info(f"Обработка категории {category} завершена.")

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Краулинг завершён за {elapsed_time:.2f} секунд.")
        logger.info(f"Все товары сохранены в каталоге '{self.output_dir}'.")

        try:
            display.stop()
        except:
            pass

    def get_alternative_search_terms(self, category: str) -> List[str]:
        """Генерирует расширенные альтернативные поисковые запросы на основе исходной категории"""
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
            "Анальные бусы": ["Анальные шарики", "Анальная цепочка с шариками", "Гирлянда анальная",
                              "Бусы для анального секса"]
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

    def create_empty_proxies_file(self):
        """Creates an empty proxies file if it doesn't exist"""
        try:
            proxies_file = "proxies.txt"  # File in the same directory as the script
            if not os.path.exists(proxies_file):
                with open(proxies_file, 'w') as f:
                    f.write("# Add your proxies here, one per line\n")
                    f.write("# Format: protocol://username:password@host:port\n")
                    f.write("# Example: http://user:pass@proxy.example.com:8080\n")
                logger.info(f"Created empty proxies file at {proxies_file}")
            else:
                logger.info(f"Proxies file already exists at {proxies_file}")
        except Exception as e:
            logger.error(f"Error creating proxies file: {e}")

    def load_proxies(self):
        """Loads proxies from the proxies.txt file"""
        proxies = []
        proxies_file = "proxies.txt"

        if os.path.exists(proxies_file):
            try:
                with open(proxies_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            proxies.append(line)
                logger.info(f"Loaded {len(proxies)} proxies from {proxies_file}")
            except Exception as e:
                logger.error(f"Error loading proxies: {e}")
        else:
            logger.warning(f"Proxies file {proxies_file} not found. Creating empty file.")
            self.create_empty_proxies_file()

        return proxies

    def find_free_proxies(self, count=10):
        """
        Find free proxies from public sources

        Args:
            count: Maximum number of proxies to find

        Returns:
            List of proxy strings in format protocol://host:port
        """
        import requests
        from bs4 import BeautifulSoup

        found_proxies = []
        logger.info("Searching for free proxies...")

        # Try multiple free proxy sources
        try:
            # Source 1: https://free-proxy-list.net/
            response = requests.get('https://free-proxy-list.net/', timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', attrs={'class': 'table table-striped table-bordered'})

            if table:
                for row in table.find_all('tr')[1:]:  # Skip header row
                    columns = row.find_all('td')
                    if len(columns) >= 7:
                        ip = columns[0].text.strip()
                        port = columns[1].text.strip()
                        https = columns[6].text.strip() == 'yes'
                        protocol = 'https' if https else 'http'

                        proxy = f"{protocol}://{ip}:{port}"
                        found_proxies.append(proxy)

                        if len(found_proxies) >= count:
                            break

            logger.info(f"Found {len(found_proxies)} free proxies")
        except Exception as e:
            logger.error(f"Error finding free proxies: {e}")

        # Save found proxies to file
        if found_proxies:
            try:
                with open("proxies.txt", 'w') as f:
                    f.write("# Automatically found free proxies\n")
                    f.write("# Format: protocol://host:port\n")
                    f.write("# Note: Free proxies may be unreliable\n\n")

                    for proxy in found_proxies:
                        f.write(f"{proxy}\n")

                logger.info(f"Saved {len(found_proxies)} proxies to proxies.txt")
            except Exception as e:
                logger.error(f"Error saving proxies to file: {e}")

        return found_proxies

    def test_proxies(self, max_to_test=10, timeout=5):
        """
        Test proxies to find working ones

        Args:
            max_to_test: Maximum number of proxies to test
            timeout: Connection timeout in seconds

        Returns:
            List of working proxies
        """
        import requests
        import concurrent.futures

        logger.info(f"Testing {min(len(self.proxies), max_to_test)} proxies...")
        working_proxies = []

        def test_proxy(proxy):
            try:
                response = requests.get(
                    'https://www.google.com',
                    proxies={'http': proxy, 'https': proxy},
                    timeout=timeout
                )
                if response.status_code == 200:
                    logger.info(f"Proxy works: {proxy}")
                    return proxy
            except:
                pass
            return None

        # Test proxies in parallel
        proxies_to_test = self.proxies[:max_to_test]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(proxies_to_test))) as executor:
            futures = {executor.submit(test_proxy, proxy): proxy for proxy in proxies_to_test}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    working_proxies.append(result)

        logger.info(f"Found {len(working_proxies)} working proxies out of {len(proxies_to_test)} tested")

        # Save working proxies to a separate file
        if working_proxies:
            try:
                with open("working_proxies.txt", 'w') as f:
                    f.write("# Tested working proxies\n")
                    for proxy in working_proxies:
                        f.write(f"{proxy}\n")
            except Exception as e:
                logger.error(f"Error saving working proxies: {e}")

        # Update the class proxies list
        self.proxies = working_proxies
        return working_proxies

    def get_next_proxy(self):
        """
        Get the next proxy from the rotation
        Returns: A proxy URL or None if no proxies available
        """
        if not self.proxies:
            return None

        # Simple round-robin proxy selection
        proxy = self.proxies.pop(0)
        self.proxies.append(proxy)  # Put it back at the end

        return proxy

    def use_new_proxy(self, driver):
        """
        Close the current driver and create a new one with a different proxy
        Returns: New WebDriver instance
        """
        try:
            driver.quit()
        except:
            pass

        # Get a different proxy
        proxy = self.get_next_proxy()
        logger.info(f"Switching to new proxy: {proxy}")

        # Create a new driver with this proxy
        return self.get_driver()

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

    crawler = WildberriesCrawler(category_targets_needed_scaled, max_workers=12)

    if not os.path.exists("proxies.txt"):
        crawler.create_empty_proxies_file()

    crawler.run()

