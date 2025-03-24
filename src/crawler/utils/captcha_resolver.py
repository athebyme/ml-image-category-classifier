"""
Модуль для обнаружения и решения CAPTCHA.
"""
import time
import requests
from selenium.webdriver.common.by import By
from ..logging_setup import logger
from ..config import CAPTCHA_API_KEY


class CaptchaSolver:
    """
    Класс для обнаружения и решения CAPTCHA.

    Поддерживает работу с различными типами CAPTCHA, включая обычные изображения,
    reCAPTCHA и hCaptcha.
    """

    def __init__(self, api_key=None):
        """
        Инициализирует решатель CAPTCHA.

        Args:
            api_key (str, optional): API-ключ для сервиса решения CAPTCHA.
        """
        self.api_key = api_key or CAPTCHA_API_KEY

    def detect_captcha(self, driver):
        """
        Определяет наличие CAPTCHA на странице.

        Args:
            driver (WebDriver): Экземпляр веб-драйвера.

        Returns:
            bool: True, если обнаружена CAPTCHA, иначе False.
        """
        try:
            # Проверяем общие идентификаторы CAPTCHA
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
                        logger.warning("CAPTCHA обнаружена на странице!")
                        return True
                except:
                    continue

            # Также проверяем URL и исходный код на наличие маркеров CAPTCHA
            if ("/__wbaas/challenges/captcha/" in driver.current_url or
                    "__wbaas/challenges/captcha/" in driver.page_source):
                logger.warning("Обнаружен URL-адрес CAPTCHA!")
                return True

            return False
        except Exception as e:
            logger.error(f"Ошибка при проверке CAPTCHA: {e}")
            return False

    def handle_captcha(self, driver):
        """
        Пытается обработать CAPTCHA.

        Args:
            driver (WebDriver): Экземпляр веб-драйвера.

        Returns:
            bool: True, если CAPTCHA успешно обработана или не обнаружена, иначе False.
        """
        if not self.detect_captcha(driver):
            return True  # CAPTCHA не обнаружена

        try:
            # Делаем скриншот для возможного ручного вмешательства
            timestamp = int(time.time())
            screenshot_path = f"captcha_{timestamp}.png"
            driver.save_screenshot(screenshot_path)

            # Сохраняем HTML для анализа
            html_path = f"captcha_{timestamp}.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(driver.page_source)

            logger.info(f"CAPTCHA обнаружена! Скриншот сохранен в {screenshot_path}, HTML в {html_path}")

            # Пытаемся решить CAPTCHA с помощью 2captcha
            try:
                captcha_type = self.identify_captcha_type(driver)

                if captcha_type == "image":
                    # Решаем обычную CAPTCHA-изображение
                    return self.solve_image_captcha(driver, screenshot_path)
                elif captcha_type == "recaptcha":
                    # Решаем reCAPTCHA
                    return self.solve_recaptcha(driver)
                elif captcha_type == "hcaptcha":
                    # Решаем hCaptcha
                    return self.solve_hcaptcha(driver)
                else:
                    logger.warning(f"Неизвестный тип CAPTCHA: {captcha_type}")

            except ImportError:
                logger.warning("Библиотека 2captcha не установлена. Установите: pip install 2captcha-python")
            except Exception as e:
                logger.error(f"Ошибка при использовании 2captcha: {e}")

            # Проверяем режим запуска браузера для возможного ручного решения
            is_headless = "--headless" in str(driver.capabilities.get("chrome", {}).get("chromedriverArgs", []))
            if not is_headless:
                wait_time = 60  # 60 секунд для ручного решения
                logger.info(f"Ожидаем {wait_time} секунд для ручного решения CAPTCHA...")
                time.sleep(wait_time)

                # Проверяем, все еще ли мы на странице с CAPTCHA
                if not self.detect_captcha(driver):
                    logger.info("CAPTCHA, похоже, была решена вручную!")
                    return True

            # В крайнем случае: пробуем обновить страницу
            logger.warning("Не удалось решить CAPTCHA. Пробуем обновить страницу...")
            driver.refresh()
            time.sleep(5)

            return not self.detect_captcha(driver)  # Возвращаем True, если CAPTCHA исчезла после обновления

        except Exception as e:
            logger.error(f"Ошибка при попытке обработать CAPTCHA: {e}")
            return False

    def identify_captcha_type(self, driver):
        """
        Определяет тип CAPTCHA на странице.

        Args:
            driver (WebDriver): Экземпляр веб-драйвера.

        Returns:
            str: Тип CAPTCHA ("image", "recaptcha" или "hcaptcha").
        """
        try:
            # Проверяем изображение CAPTCHA (обычно используется на Wildberries)
            if driver.find_elements(By.XPATH, "//img[contains(@src, 'captcha') or contains(@class, 'captcha')]"):
                return "image"

            # Проверяем reCAPTCHA
            if (driver.find_elements(By.XPATH, "//div[contains(@class, 'g-recaptcha')]") or
                    "www.google.com/recaptcha" in driver.page_source):
                return "recaptcha"

            # Проверяем hCaptcha
            if (driver.find_elements(By.XPATH, "//div[contains(@class, 'h-captcha')]") or
                    "hcaptcha.com" in driver.page_source):
                return "hcaptcha"

            # По умолчанию для Wildberries (наиболее распространенный)
            return "image"
        except:
            return "image"  # По умолчанию, если обнаружение не удалось

    def solve_image_captcha(self, driver, screenshot_path):
        """
        Решает обычную CAPTCHA на основе изображения.

        Args:
            driver (WebDriver): Экземпляр веб-драйвера.
            screenshot_path (str): Путь к скриншоту с CAPTCHA.

        Returns:
            bool: True, если CAPTCHA была решена успешно, иначе False.
        """
        try:
            # Проверяем, установлен ли API-ключ
            if not self.api_key:
                logger.error("API-ключ 2captcha не настроен")
                return False

            # Импортируем модуль 2captcha
            from twocaptcha import TwoCaptcha, NetworkException

            # Создаем экземпляр решателя
            solver = TwoCaptcha(self.api_key)

            # Пытаемся найти элемент с изображением CAPTCHA
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

            captcha_file = screenshot_path

            if captcha_img:
                # Пытаемся получить URL изображения напрямую
                img_src = captcha_img.get_attribute('src')

                if img_src and not img_src.startswith('data:'):
                    # Загружаем изображение
                    img_response = requests.get(img_src, timeout=10)
                    if img_response.status_code == 200:
                        with open('current_captcha.png', 'wb') as f:
                            f.write(img_response.content)
                        captcha_file = 'current_captcha.png'
                else:
                    # Сохраняем скриншот только элемента CAPTCHA
                    captcha_img.screenshot('captcha_element.png')
                    captcha_file = 'captcha_element.png'

            logger.info(f"Отправляем изображение CAPTCHA в сервис 2captcha...")

            try:
                # Отправляем CAPTCHA на решение
                result = solver.normal(captcha_file)
                captcha_text = result.get('code', '')

                logger.info(f"Получен ответ от 2captcha: {captcha_text}")

                # Ищем поле ввода
                input_field = None
                try:
                    # Пробуем разные селекторы для поля ввода
                    for selector in ["input[name='captcha']", "input[id*='captcha']", "input[type='text']"]:
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            input_field = elements[0]
                            break
                except:
                    pass

                # Если нашли поле ввода, вводим решение
                if input_field:
                    input_field.clear()
                    input_field.send_keys(captcha_text)

                    # Ищем кнопку отправки
                    submit_button = None
                    try:
                        # Пробуем разные селекторы для кнопки отправки
                        for selector in ["button[type='submit']", "input[type='submit']", "button"]:
                            elements = driver.find_elements(By.CSS_SELECTOR, selector)
                            if elements:
                                submit_button = elements[0]
                                break
                    except:
                        pass

                    if submit_button:
                        submit_button.click()
                        time.sleep(3)  # Ждем обработки отправки

                        # Проверяем, осталась ли CAPTCHA
                        return not self.detect_captcha(driver)
                    else:
                        logger.warning("Не удалось найти кнопку отправки для CAPTCHA")
                else:
                    logger.warning("Не удалось найти поле ввода для CAPTCHA")

                return False  # CAPTCHA не решена

            except NetworkException as e:
                logger.error(f"Ошибка сети с 2captcha: {e}")
                return False
            except Exception as e:
                logger.error(f"Ошибка при решении CAPTCHA-изображения: {e}")
                return False
        except Exception as e:
            logger.error(f"Общая ошибка в solve_image_captcha: {e}")
            return False

    def solve_recaptcha(self, driver):
        """
        Решает Google reCAPTCHA.

        Args:
            driver (WebDriver): Экземпляр веб-драйвера.

        Returns:
            bool: True, если CAPTCHA была решена успешно, иначе False.
        """
        try:
            # Проверяем, установлен ли API-ключ
            if not self.api_key:
                logger.error("API-ключ 2captcha не настроен")
                return False

            # Импортируем модуль 2captcha
            from twocaptcha import TwoCaptcha
            import re

            # Создаем экземпляр решателя
            solver = TwoCaptcha(self.api_key)

            # Получаем ключ сайта
            site_key = None
            try:
                # Пытаемся найти ключ сайта в элементе recaptcha
                recaptcha_div = driver.find_element(By.CSS_SELECTOR, ".g-recaptcha")
                site_key = recaptcha_div.get_attribute("data-sitekey")
            except:
                # Пытаемся извлечь из исходного кода страницы
                match = re.search(r'data-sitekey=["\']([^"\']+)["\']', driver.page_source)
                if match:
                    site_key = match.group(1)

            if not site_key:
                logger.error("Не удалось найти ключ сайта reCAPTCHA")
                return False

            logger.info(f"Найден ключ сайта reCAPTCHA: {site_key}")

            # Отправляем CAPTCHA на решение
            result = solver.recaptcha(
                sitekey=site_key,
                url=driver.current_url
            )

            g_response = result.get('code')
            logger.info("Получено решение reCAPTCHA от 2captcha")

            # Применяем решение
            driver.execute_script(f"document.getElementById('g-recaptcha-response').innerHTML = '{g_response}';")

            # Отправляем форму
            try:
                submit_buttons = driver.find_elements(By.XPATH,
                                                      "//button[@type='submit'] | //input[@type='submit'] | //button[contains(text(), 'Submit')]")
                if submit_buttons:
                    submit_buttons[0].click()
                else:
                    # Пытаемся вызвать отправку формы через JavaScript
                    driver.execute_script("""
                        var forms = document.getElementsByTagName('form');
                        if (forms.length > 0) {
                            forms[0].submit();
                        }
                    """)
            except Exception as e:
                logger.warning(f"Ошибка при отправке формы reCAPTCHA: {e}")

            time.sleep(3)  # Ждем обработки отправки

            # Проверяем, осталась ли CAPTCHA
            return not self.detect_captcha(driver)

        except Exception as e:
            logger.error(f"Ошибка при решении reCAPTCHA: {e}")
            return False

    def solve_hcaptcha(self, driver):
        """
        Решает hCaptcha.

        Args:
            driver (WebDriver): Экземпляр веб-драйвера.

        Returns:
            bool: True, если CAPTCHA была решена успешно, иначе False.
        """
        try:
            # Проверяем, установлен ли API-ключ
            if not self.api_key:
                logger.error("API-ключ 2captcha не настроен")
                return False

            # Импортируем модуль 2captcha
            from twocaptcha import TwoCaptcha
            import re

            # Создаем экземпляр решателя
            solver = TwoCaptcha(self.api_key)

            # Получаем ключ сайта
            site_key = None
            try:
                # Пытаемся найти ключ сайта в элементе hcaptcha
                hcaptcha_div = driver.find_element(By.CSS_SELECTOR, ".h-captcha")
                site_key = hcaptcha_div.get_attribute("data-sitekey")
            except:
                # Пытаемся извлечь из исходного кода страницы
                match = re.search(r'data-sitekey=["\']([^"\']+)["\']', driver.page_source)
                if match:
                    site_key = match.group(1)

            if not site_key:
                logger.error("Не удалось найти ключ сайта hCaptcha")
                return False

            logger.info(f"Найден ключ сайта hCaptcha: {site_key}")

            # Отправляем CAPTCHA на решение
            result = solver.hcaptcha(
                sitekey=site_key,
                url=driver.current_url
            )

            h_response = result.get('code')
            logger.info("Получено решение hCaptcha от 2captcha")

            # Применяем решение
            driver.execute_script(f"document.getElementsByName('h-captcha-response')[0].innerHTML = '{h_response}';")

            # Отправляем форму
            try:
                submit_buttons = driver.find_elements(By.XPATH,
                                                      "//button[@type='submit'] | //input[@type='submit'] | //button[contains(text(), 'Submit')]")
                if submit_buttons:
                    submit_buttons[0].click()
                else:
                    # Пытаемся вызвать отправку формы через JavaScript
                    driver.execute_script("""
                        var forms = document.getElementsByTagName('form');
                        if (forms.length > 0) {
                            forms[0].submit();
                        }
                    """)
            except Exception as e:
                logger.warning(f"Ошибка при отправке формы hCaptcha: {e}")

            time.sleep(3)  # Ждем обработки отправки

            # Проверяем, осталась ли CAPTCHA
            return not self.detect_captcha(driver)

        except Exception as e:
            logger.error(f"Ошибка при решении hCaptcha: {e}")
            return False