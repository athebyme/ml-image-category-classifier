# Crawler для Wildberries и Ozon

Модульный и масштабируемый краулер для сбора данных о товарах с сайтов Wildberries и Ozon.

## Особенности

- Поддержка нескольких источников данных (Wildberries и Ozon)
- Модульная архитектура с разделением ответственности
- Мультипоточная обработка для повышения производительности
- Обход защит от ботов и CAPTCHA
- Экспоненциальная задержка для избежания блокировок
- Поддержка прокси
- Детальное логирование процесса
- Умное определение URLs изображений высокого качества
- Обработчики сигналов для корректного завершения работы

## Структура проекта

```
crawler/
    __init__.py
    config.py                         # Настройки и конфигурации
    logging_setup.py                  # Настройка логирования
    utils/
        __init__.py
        backoff.py                    # Класс ExponentialBackoff
        proxy_manager.py              # Управление прокси
        browser.py                    # Создание и настройка веб-драйверов
        captcha_solver.py             # Обработка CAPTCHA
    parsers/
        __init__.py
        wildberries_product_parser.py # Парсер страниц товаров Wildberries
        wildberries_search_parser.py  # Парсер страниц поиска Wildberries
        ozon_product_parser.py        # Парсер страниц товаров Ozon
        ozon_search_parser.py         # Парсер страниц поиска Ozon
    storage/
        __init__.py
        json_storage.py               # Работа с JSON данными
    wildberries_crawler.py            # Основной класс краулера Wildberries
    ozon_crawler.py                   # Основной класс краулера Ozon
    main.py                           # Точка входа
```

## Требования

- Python 3.7+
- Установленный Google Chrome или Chromium
- Необходимые библиотеки (см. `requirements.txt`)

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/athebyme/crawler.git
   cd crawler
   ```

2. Установите необходимые зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Создайте файл `categories.json` или используйте существующий.

## Использование

### Базовый запуск:

```bash
# Для Wildberries (по умолчанию)
python -m crawler.main

# Для Ozon
python -m crawler.main --source ozon
```

### Запуск с параметрами:

```bash
# Для Wildberries с дополнительными параметрами
python -m crawler.main --source wildberries --max-workers 12 --categories-file my_categories.json --scale-factor 0.5

# Для Ozon с дополнительными параметрами
python -m crawler.main --source ozon --max-workers 8 --categories-file ozon_categories.json --output-dir ozon_data
```

### Параметры:

- `--source`: Источник данных (wildberries или ozon, по умолчанию: wildberries)
- `--max-workers`: Максимальное количество рабочих потоков (по умолчанию: 8)
- `--categories-file`: Путь к JSON-файлу со списком категорий (по умолчанию: categories.json)
- `--output-dir`: Директория для сохранения результатов (по умолчанию: ~/shared_crawler_output)
- `--scale-factor`: Коэффициент масштабирования для целевых значений категорий (по умолчанию: 1.0)

## Файл категорий

Файл `categories.json` содержит словарь с категориями и целевым количеством товаров для сбора:

```json
{
  "Категория1": 100,
  "Категория2": 200,
  "Категория3": 50
}
```

## Прокси

Для использования прокси-серверов, создайте файл `proxies.txt` в корневом каталоге проекта:

```
# Формат: protocol://username:password@host:port
http://user:pass@proxy.example.com:8080
http://123.123.123.123:8080
```

## Использование API для CAPTCHA

Для автоматического решения CAPTCHA, настройте переменную окружения `CAPTCHA_API_KEY` с вашим ключом от сервиса [2Captcha](https://2captcha.com/):

```bash
export CAPTCHA_API_KEY="ваш_api_ключ"
```

## Примеры кода

### Использование WildberriesCrawler напрямую:

```python
from crawler.wildberries_crawler import WildberriesCrawler

# Настройка целевых категорий
category_targets = {
    "Вибраторы": 100,
    "Презервативы": 50
}

# Создание и запуск краулера Wildberries
crawler = WildberriesCrawler(category_targets, max_workers=8, output_dir="wb_data")
crawler.run()
```

### Использование OzonCrawler напрямую:

```python
from crawler.ozon_crawler import OzonCrawler

# Настройка целевых категорий
category_targets = {
    "Вибраторы": 100,
    "Презервативы": 50
}

# Создание и запуск краулера Ozon
crawler = OzonCrawler(category_targets, max_workers=8, output_dir="ozon_data")
crawler.run()
```

### Использование WildberriesSearchParser для сбора только URL с Wildberries:

```python
from crawler.utils.browser import BrowserManager
from crawler.parsers.wildberries_search_parser import WildberriesSearchParser

browser_manager = BrowserManager()
search_parser = WildberriesSearchParser(browser_manager)

# Получение URL-адресов товаров с Wildberries
product_urls = search_parser.collect_product_urls("Вибраторы", target_count=10)
print(f"Найдено {len(product_urls)} URL-адресов товаров на Wildberries")
```

### Использование OzonSearchParser для сбора только URL с Ozon:

```python
from crawler.utils.browser import BrowserManager
from crawler.parsers.ozon_search_parser import OzonSearchParser

browser_manager = BrowserManager()
search_parser = OzonSearchParser(browser_manager)

# Получение URL-адресов товаров с Ozon
product_urls = search_parser.collect_product_urls("Вибраторы", target_count=10)
print(f"Найдено {len(product_urls)} URL-адресов товаров на Ozon")
```

### Использование WildberriesProductParser для обработки одного товара с Wildberries:

```python
from crawler.utils.browser import BrowserManager
from crawler.parsers.wildberries_product_parser import WildberriesProductParser

browser_manager = BrowserManager()
product_parser = WildberriesProductParser(browser_manager)

# Обработка страницы товара Wildberries
driver = browser_manager.get_driver()
try:
    url = "https://www.wildberries.ru/catalog/12345678/detail.aspx"
    product_data = product_parser.process_product_page(driver, url, "Тестовая категория")
    print(product_data)
finally:
    driver.quit()
```

### Использование OzonProductParser для обработки одного товара с Ozon:

```python
from crawler.utils.browser import BrowserManager
from crawler.parsers.ozon_product_parser import OzonProductParser

browser_manager = BrowserManager()
product_parser = OzonProductParser(browser_manager)

# Обработка страницы товара Ozon
driver = browser_manager.get_driver()
try:
    url = "https://www.ozon.ru/product/massazhnyy-apparat-12345678/"
    product_data = product_parser.process_product_page(driver, url, "Тестовая категория")
    print(product_data)
finally:
    driver.quit()
```

## Логирование

Логи сохраняются в файл `crawler.log` с ротацией по размеру (5 МБ).

## Лицензия

MIT