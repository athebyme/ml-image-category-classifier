"""
Пакет утилит для краулера Wildberries.

Включает в себя классы и функции для:
- Управления браузером и WebDriver
- Работы с прокси-серверами
- Решения CAPTCHA
- Реализации экспоненциальной задержки
"""

from crawler.utils.browser import BrowserManager
from crawler.utils.proxy_manager import ProxyManager
from crawler.utils.captcha_solver import CaptchaSolver
from crawler.utils.backoff import ExponentialBackoff

__all__ = [
    'BrowserManager',
    'ProxyManager',
    'CaptchaSolver',
    'ExponentialBackoff'
]