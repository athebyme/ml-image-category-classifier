"""
Module for detecting and solving CAPTCHAs using free methods.
"""
import time
import random
import cv2
from selenium.webdriver.common.by import By
from ..logging_setup import logger


class SimpleCaptchaSolver:
    """
    Class for detecting and solving CAPTCHAs using free methods.

    Uses OpenCV and Tesseract OCR for basic image CAPTCHA solving,
    and implements strategies to avoid triggering CAPTCHAs.
    """

    def __init__(self, tesseract_path=None):
        """
        Initializes the CAPTCHA solver.

        Args:
            tesseract_path (str, optional): Path to Tesseract executable.
        """
        self.tesseract_path = tesseract_path
        # Try to import pytesseract
        try:
            import pytesseract
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            self.ocr_available = True
            logger.info("Tesseract OCR initialized for CAPTCHA solving")
        except ImportError:
            self.ocr_available = False
            logger.warning("pytesseract not available. Install with: pip install pytesseract")
            logger.warning("You also need to install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")

    def detect_captcha(self, driver):
        """
        Detects if a CAPTCHA is present on the page.

        Args:
            driver (WebDriver): Selenium WebDriver instance.

        Returns:
            bool: True if CAPTCHA is detected, False otherwise.
        """
        try:
            # Check common CAPTCHA identifiers
            captcha_selectors = [
                (By.ID, "__wbaas_captcha_container"),
                (By.XPATH, "//div[contains(@class, 'captcha')]"),
                (By.XPATH, "//title[contains(text(), 'Почти готово')]"),
                (By.XPATH, "//p[contains(text(), 'IP-адрес')]"),
                (By.XPATH, "//img[contains(@src, 'captcha')]"),
                (By.XPATH, "//div[contains(@class, 'g-recaptcha')]"),
                (By.XPATH, "//iframe[contains(@src, 'recaptcha')]"),
            ]

            for selector_type, selector in captcha_selectors:
                try:
                    element = driver.find_element(selector_type, selector)
                    if element and element.is_displayed():
                        logger.warning("CAPTCHA detected on page!")
                        return True
                except:
                    continue

            # Check URL and page source for CAPTCHA markers
            captcha_markers = [
                "/__wbaas/challenges/captcha/",
                "captcha",
                "recaptcha",
                "hcaptcha",
                "challenge",
                "bot-detection"
            ]

            for marker in captcha_markers:
                if marker in driver.current_url.lower() or marker in driver.page_source.lower():
                    logger.warning(f"CAPTCHA marker '{marker}' detected!")
                    return True

            return False
        except Exception as e:
            logger.error(f"Error checking for CAPTCHA: {e}")
            return False

    def handle_captcha(self, driver):
        """
        Attempts to handle a CAPTCHA.

        Args:
            driver (WebDriver): Selenium WebDriver instance.

        Returns:
            bool: True if CAPTCHA was successfully handled or not detected, False otherwise.
        """
        if not self.detect_captcha(driver):
            return True  # No CAPTCHA detected

        try:
            # Take screenshot for possible manual intervention
            timestamp = int(time.time())
            screenshot_path = f"captcha_{timestamp}.png"
            driver.save_screenshot(screenshot_path)
            logger.info(f"CAPTCHA detected! Screenshot saved to {screenshot_path}")

            # If Tesseract OCR is available, try to solve image CAPTCHA
            if self.ocr_available and self._is_image_captcha(driver):
                return self._solve_image_captcha(driver)

            # Try rotating user agent
            self._rotate_user_agent(driver)

            # Try random delay
            delay = random.uniform(5, 15)
            logger.info(f"CAPTCHA detected, waiting {delay:.2f} seconds...")
            time.sleep(delay)

            # Check if we're in headless mode
            is_headless = "--headless" in str(driver.capabilities.get("chrome", {}).get("chromedriverArgs", []))
            if not is_headless:
                # Wait for manual solving in visible browser mode
                wait_time = 60
                logger.info(f"Waiting {wait_time} seconds for manual CAPTCHA solution...")
                time.sleep(wait_time)

                # Check if CAPTCHA is still present
                if not self.detect_captcha(driver):
                    logger.info("CAPTCHA appears to have been solved manually!")
                    return True

            # Try refreshing the page as a last resort
            logger.warning("Failed to solve CAPTCHA. Trying to refresh page...")
            driver.refresh()
            time.sleep(5)

            return not self.detect_captcha(driver)

        except Exception as e:
            logger.error(f"Error handling CAPTCHA: {e}")
            return False

    def _is_image_captcha(self, driver):
        """
        Detects if the CAPTCHA is a simple image type.

        Args:
            driver (WebDriver): Selenium WebDriver instance.

        Returns:
            bool: True if an image CAPTCHA is detected, False otherwise.
        """
        try:
            # Look for image elements that might contain a CAPTCHA
            captcha_img_selectors = [
                "//img[contains(@src, 'captcha')]",
                "//div[contains(@class, 'captcha')]//img",
                "//div[@id='__wbaas_captcha_container']//img"
            ]

            for selector in captcha_img_selectors:
                elements = driver.find_elements(By.XPATH, selector)
                if elements and elements[0].is_displayed():
                    return True

            return False
        except:
            return False

    def _solve_image_captcha(self, driver):
        """
        Attempts to solve an image CAPTCHA using OCR.

        Args:
            driver (WebDriver): Selenium WebDriver instance.

        Returns:
            bool: True if CAPTCHA was solved, False otherwise.
        """
        try:
            import pytesseract
            from PIL import Image, ImageEnhance, ImageFilter

            # Find the CAPTCHA image
            captcha_img = None
            captcha_img_selectors = [
                "//img[contains(@src, 'captcha')]",
                "//div[contains(@class, 'captcha')]//img",
                "//div[@id='__wbaas_captcha_container']//img"
            ]

            for selector in captcha_img_selectors:
                elements = driver.find_elements(By.XPATH, selector)
                if elements and elements[0].is_displayed():
                    captcha_img = elements[0]
                    break

            if not captcha_img:
                logger.warning("Could not find CAPTCHA image")
                return False

            # Save the CAPTCHA image
            captcha_img.screenshot('captcha_element.png')

            # Preprocess the image to improve OCR accuracy
            img = Image.open('captcha_element.png')

            # Convert to grayscale
            img = img.convert('L')

            # Increase contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2)

            # Apply threshold to make text more distinct
            threshold = 150
            img = img.point(lambda p: 255 if p > threshold else 0)

            # Apply noise reduction
            img = img.filter(ImageFilter.MedianFilter())

            # Save preprocessed image
            img.save('captcha_processed.png')

            # Use OCR to extract text
            captcha_text = pytesseract.image_to_string(img,
                                                       config='--psm 7 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
            captcha_text = captcha_text.strip()

            logger.info(f"OCR detected CAPTCHA text: {captcha_text}")

            if not captcha_text:
                logger.warning("OCR could not recognize any text in CAPTCHA")
                return False

            # Find the input field
            input_field = None
            for selector in ["input[name='captcha']", "input[id*='captcha']", "input[type='text']"]:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    input_field = elements[0]
                    break

            if not input_field:
                logger.warning("Could not find CAPTCHA input field")
                return False

            # Enter the CAPTCHA text
            input_field.clear()
            input_field.send_keys(captcha_text)

            # Find and click the submit button
            submit_button = None
            for selector in ["button[type='submit']", "input[type='submit']", "button"]:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    submit_button = elements[0]
                    break

            if not submit_button:
                logger.warning("Could not find submit button")
                return False

            submit_button.click()
            time.sleep(3)

            # Check if CAPTCHA is still present
            return not self.detect_captcha(driver)

        except Exception as e:
            logger.error(f"Error solving image CAPTCHA: {e}")
            return False

    def _rotate_user_agent(self, driver):
        """
        Rotates the user agent to potentially bypass CAPTCHA.

        Args:
            driver (WebDriver): Selenium WebDriver instance.
        """
        try:
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59"
            ]

            # Set a random user agent
            new_agent = random.choice(user_agents)
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": new_agent})
            logger.info(f"Rotated user agent to: {new_agent}")
        except Exception as e:
            logger.warning(f"Failed to rotate user agent: {e}")