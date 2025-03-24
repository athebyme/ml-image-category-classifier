"""
Пакет хранилища данных для краулера Wildberries.

Включает в себя классы для:
- Хранения данных в формате JSON
- Управления сохраненными товарами
"""

from ..storage.json_storage import JsonStorage

__all__ = ['JsonStorage']