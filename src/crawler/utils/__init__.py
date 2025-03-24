"""
Пакет утилит для краулера Wildberries.

Включает в себя классы и функции для:
- Управления браузером и WebDriver
- Работы с прокси-серверами
- Решения CAPTCHA
- Реализации экспоненциальной задержки
"""

from ..utils.browser import BrowserManager
from ..utils.proxy_manager import ProxyManager
from ..utils.captcha_resolver import CaptchaSolver
from ..utils.backoff import ExponentialBackoff

__all__ = [
    'BrowserManager',
    'ProxyManager',
    'CaptchaSolver',
    'ExponentialBackoff'
]