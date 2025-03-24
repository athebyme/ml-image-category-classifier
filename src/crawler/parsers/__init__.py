"""
Пакет парсеров для краулера.

Включает в себя классы для парсинга:
- Страниц товаров Wildberries и Ozon
- Страниц поиска и категорий
"""

from .wildberries_product_parser import WildberriesProductParser
from .wildberries_search_parser import WildberriesSearchParser
from .ozon_product_parser import OzonProductParser
from .ozon_search_parser import OzonSearchParser

__all__ = [
    'WildberriesProductParser',
    'WildberriesSearchParser',
    'OzonProductParser',
    'OzonSearchParser'
]