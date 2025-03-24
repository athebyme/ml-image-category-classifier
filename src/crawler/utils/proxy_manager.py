"""
Модуль для управления прокси-серверами.
"""
import os
import requests
import concurrent.futures
from bs4 import BeautifulSoup
from ..logging_setup import logger
from ..config import PROXIES_FILE


class ProxyManager:
    """
    Класс для управления прокси-серверами.

    Позволяет загружать, тестировать, находить и ротировать прокси-серверы.
    """

    def __init__(self):
        """
        Инициализирует менеджер прокси.
        """
        self.proxies = []
        self.proxy_index = 0
        self._load_proxies()

    def _load_proxies(self):
        """
        Загружает список прокси из файла.

        Returns:
            bool: True, если прокси были загружены успешно, иначе False.
        """
        if os.path.exists(PROXIES_FILE):
            try:
                with open(PROXIES_FILE, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            self.proxies.append(line)

                logger.info(f"Загружено {len(self.proxies)} прокси из {PROXIES_FILE}")
                return True
            except Exception as e:
                logger.error(f"Ошибка при загрузке прокси: {e}")
                return False
        else:
            logger.warning(f"Файл прокси {PROXIES_FILE} не найден. Создаем пустой файл.")
            self._create_empty_proxies_file()
            return False

    def _create_empty_proxies_file(self):
        """
        Создает пустой файл proxies.txt с комментариями о формате.
        """
        try:
            with open(PROXIES_FILE, "w", encoding="utf-8") as f:
                f.write("# Файл для списка прокси\n")
                f.write("# Формат: ip:port или user:pass@ip:port\n")
                f.write("# Примеры:\n")
                f.write("# 192.168.1.1:8080\n")
                f.write("# username:password@192.168.1.1:8080\n")
            logger.info(f"Создан пустой файл прокси {PROXIES_FILE}")
        except Exception as e:
            logger.error(f"Ошибка при создании файла с прокси: {e}")

    def find_free_proxies(self, count=10):
        """
        Ищет бесплатные прокси из публичных источников.

        Args:
            count (int): Максимальное количество прокси для поиска.

        Returns:
            list: Список найденных прокси в формате protocol://host:port.
        """
        found_proxies = []
        logger.info("Поиск бесплатных прокси...")

        try:
            # Источник 1: https://free-proxy-list.net/
            response = requests.get('https://free-proxy-list.net/', timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', attrs={'class': 'table table-striped table-bordered'})

            if table:
                for row in table.find_all('tr')[1:]:  # Пропускаем заголовок
                    columns = row.find_all('td')
                    if len(columns) >= 7:
                        ip = columns[0].text.strip()
                        port = columns[1].text.strip()
                        https = columns[6].text.strip() == 'yes'
                        protocol = 'https' if https else 'http'

                        proxy = f"{protocol}://{ip}:{port}"
                        found_proxies.append(proxy)

                        if len(found_proxies) >= count:
                            break

            logger.info(f"Найдено {len(found_proxies)} бесплатных прокси")
        except Exception as e:
            logger.error(f"Ошибка при поиске бесплатных прокси: {e}")

        # Сохраняем найденные прокси в файл
        if found_proxies:
            try:
                with open(PROXIES_FILE, 'w') as f:
                    f.write("# Автоматически найденные бесплатные прокси\n")
                    f.write("# Формат: protocol://host:port\n")
                    f.write("# Примечание: Бесплатные прокси могут быть ненадежными\n\n")

                    for proxy in found_proxies:
                        f.write(f"{proxy}\n")

                logger.info(f"Сохранено {len(found_proxies)} прокси в {PROXIES_FILE}")

                # Обновляем список прокси
                self.proxies = found_proxies
            except Exception as e:
                logger.error(f"Ошибка при сохранении прокси в файл: {e}")

        return found_proxies

    def test_proxies(self, max_to_test=10, timeout=5):
        """
        Тестирует прокси для поиска работающих.

        Args:
            max_to_test (int): Максимальное количество прокси для тестирования.
            timeout (int): Таймаут соединения в секундах.

        Returns:
            list: Список работающих прокси.
        """
        if not self.proxies:
            logger.warning("Нет прокси для тестирования.")
            return []

        logger.info(f"Тестирование {min(len(self.proxies), max_to_test)} прокси...")
        working_proxies = []

        def test_proxy(proxy):
            try:
                response = requests.get(
                    'https://www.google.com',
                    proxies={'http': proxy, 'https': proxy},
                    timeout=timeout
                )
                if response.status_code == 200:
                    logger.info(f"Прокси работает: {proxy}")
                    return proxy
            except:
                pass
            return None

        # Тестируем прокси параллельно
        proxies_to_test = self.proxies[:max_to_test]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(proxies_to_test))) as executor:
            futures = {executor.submit(test_proxy, proxy): proxy for proxy in proxies_to_test}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    working_proxies.append(result)

        logger.info(f"Найдено {len(working_proxies)} работающих прокси из {len(proxies_to_test)} проверенных")

        # Сохраняем работающие прокси в отдельный файл
        if working_proxies:
            try:
                with open("working_proxies.txt", 'w') as f:
                    f.write("# Проверенные работающие прокси\n")
                    for proxy in working_proxies:
                        f.write(f"{proxy}\n")
            except Exception as e:
                logger.error(f"Ошибка при сохранении работающих прокси: {e}")

        # Обновляем список прокси
        self.proxies = working_proxies
        return working_proxies

    def get_next_proxy(self):
        """
        Получает следующий прокси из ротации.

        Returns:
            str: URL прокси или None, если нет доступных прокси.
        """
        if not self.proxies:
            return None

        # Простая ротация прокси по кругу
        proxy = self.proxies[self.proxy_index]
        self.proxy_index = (self.proxy_index + 1) % len(self.proxies)

        return proxy

    def get_random_proxy(self):
        """
        Получает случайный прокси из списка.

        Returns:
            str: URL прокси или None, если нет доступных прокси.
        """
        if not self.proxies:
            return None

        return random.choice(self.proxies)