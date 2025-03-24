"""
Модуль конфигурации для краулера Wildberries.
"""
import os

from selenium.webdriver.common.by import By

# Базовые директории
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "shared_crawler_output")
COOKIES_DIR = os.path.join(BASE_DIR, "cookies")
PROXIES_FILE = os.path.join(BASE_DIR, "proxies.txt")

# Настройки по умолчанию
DEFAULT_MAX_WORKERS = 8
MAX_RETRIES = 3
MAX_CONSECUTIVE_FAILURES = 3
MAX_PAGES_PER_TERM = 50
MAX_REQUESTS_PER_SESSION = (15, 25)  # мин. и макс. значения для рандомизации

# Настройки задержек
EXPONENTIAL_BACKOFF = {
    "initial_delay": 5,
    "max_delay": 300,
    "factor": 2
}

# Настройки сети и браузера
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15"
]

SCREEN_RESOLUTIONS = [
    (1366, 768),
    (1920, 1080),
    (1440, 900),
    (1536, 864)
]

LANGUAGES = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.9",
    "ru-RU,ru;q=0.9",
    "de-DE,de;q=0.9"
]

# API ключи для сервисов
CAPTCHA_API_KEY = os.environ.get('CAPTCHA_API_KEY', '')

# Селекторы для парсинга
SELECTORS = {
    "product_cards": [
        ".product-card__wrapper",
        ".product-card",
        "a.j-card-link",
        "[data-card-index]",
        ".catalog-page .product-card"
    ],
    "product_links": [
        "a.product-card__link.j-card-link.j-open-full-product-card",
        "a.product-card__main.j-card-link",
        "a[href*='/catalog/'][href*='/detail.aspx']"
    ],
    "captcha": [
        (By.ID, "__wbaas_captcha_container"),
        (By.XPATH, "//div[contains(@class, 'captcha')]"),
        (By.XPATH, "//title[contains(text(), 'Почти готово')]"),
        (By.XPATH, "//p[contains(text(), 'IP-адрес')]")
    ]
}

# URLS
BASE_SEARCH_URL = "https://www.wildberries.ru/catalog/0/search.aspx?search={}&page={}"

# Инициализируем директории
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COOKIES_DIR, exist_ok=True)