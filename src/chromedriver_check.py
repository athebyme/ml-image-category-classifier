from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Точный путь к ChromeDriver
service = Service('/usr/local/bin/chromedriver/chromedriver')

# Опции Chrome
chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--headless')  # Если нужен режим без графического интерфейса

try:
    # Создание драйвера с явным указанием пути
    driver = webdriver.Chrome(service=service, options=chrome_options)
    print("Драйвер успешно создан!")

    # Тестовый переход
    driver.get('https://www.example.com')
    print("Успешный переход на страницу")

    driver.quit()
except Exception as e:
    print(f"Ошибка при создании драйвера: {e}")