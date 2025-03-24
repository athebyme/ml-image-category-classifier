"""
Пакет утилит для краулера Wildberries.

Включает в себя классы и функции для:
- Управления браузером и WebDriver
- Работы с прокси-серверами
- Решения CAPTCHA
- Реализации экспоненциальной задержки
"""

from .browser import BrowserManager
from .proxy_manager import ProxyManager
from .captcha_resolver import CaptchaSolver
from .backoff import ExponentialBackoff

__all__ = [
    'BrowserManager',
    'ProxyManager',
    'CaptchaSolver',
    'ExponentialBackoff'
]