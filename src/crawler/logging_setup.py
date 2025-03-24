"""
Модуль настройки логирования для краулера Wildberries.
"""
import os
from loguru import logger
from pyvirtualdisplay import Display


def setup_logging(log_file="crawler.log"):
    """
    Настраивает логирование для краулера.

    Args:
        log_file (str): Путь к файлу логов.

    Returns:
        logger: Настроенный объект логгера.
    """
    # Удаляем дефолтные обработчики
    logger.remove()

    # Добавляем логирование в файл с ротацией
    logger.add(
        log_file,
        format="{time} {level} {message}",
        level="INFO",
        rotation="5 MB"
    )

    return logger


def setup_virtual_display(visible=0, size=(1920, 1080)):
    """
    Настраивает виртуальный дисплей для запуска браузера в headless режиме.

    Args:
        visible (int): Видимость дисплея (0 - скрытый).
        size (tuple): Размер дисплея (ширина, высота).

    Returns:
        Display: Объект виртуального дисплея.
    """
    display = Display(visible=visible, size=size)
    display.start()

    return display


# Инициализация логгера при импорте модуля
logger = setup_logging()