"""
Модуль для реализации экспоненциальной задержки при ошибках и повторных запросах.
"""
import random
from ..logging_setup import logger


class ExponentialBackoff:
    """
    Класс для реализации экспоненциальной задержки при повторных запросах.

    Увеличивает время ожидания между запросами по экспоненте в случае ошибок,
    чтобы избежать блокировки и не перегружать сервер.
    """

    def __init__(self, initial_delay=5, max_delay=300, factor=2):
        """
        Инициализирует экспоненциальную задержку.

        Args:
            initial_delay (int): Начальная задержка в секундах.
            max_delay (int): Максимальная задержка в секундах.
            factor (int): Множитель для увеличения задержки с каждой попыткой.
        """
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.factor = factor
        self.attempt = 0

    def reset(self):
        """
        Сбрасывает счетчик попыток.
        """
        self.attempt = 0

    def delay(self):
        """
        Рассчитывает время задержки для текущей попытки.

        Returns:
            float: Время задержки в секундах.
        """
        wait_time = min(self.initial_delay * (self.factor ** self.attempt), self.max_delay)
        self.attempt += 1

        # Добавляем случайное отклонение ±20% для маскировки автоматизации
        jitter = random.uniform(0.8, 1.2)

        return wait_time * jitter

    def wait(self):
        """
        Выполняет задержку с обратным отсчетом в логах.

        Returns:
            float: Фактическое время задержки в секундах.
        """
        import time
        actual_delay = self.delay()

        logger.warning(
            f"Применяется экспоненциальная задержка: {actual_delay:.2f} секунд (попытка {self.attempt})"
        )

        # Пауза с обратным отсчетом для отображения в логах
        for i in range(int(actual_delay), 0, -10):
            remaining = min(i, 10)  # Выводим каждые 10 секунд или меньше
            if i <= 30 or i % 30 == 0:  # Сокращаем количество записей в лог
                logger.debug(f"Осталось ждать: {i} секунд")
            time.sleep(remaining)

        # Сбрасываем счетчик попыток после длительной паузы
        if actual_delay > 60:
            self.reset()
            logger.info("Счетчик попыток сброшен после длительной паузы")

        return actual_delay