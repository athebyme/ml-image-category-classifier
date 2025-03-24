"""
Пакет парсеров для краулера Wildberries.

Включает в себя классы для парсинга:
- Страниц товаров
- Страниц поиска и категорий
"""

from crawler.parsers.product_parser import ProductParser
from crawler.parsers.search_parser import SearchParser

__all__ = ['ProductParser', 'SearchParser']