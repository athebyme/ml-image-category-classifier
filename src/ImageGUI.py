import gc
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk
import requests
import json
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\atheb\PycharmProjects\image-athebyme")

class ImageLabeler:
    def __init__(self, image_dir, categories):
        # Используем абсолютный путь от корня проекта, добавляя 'src'
        self.image_dir = PROJECT_DIR / 'src' / image_dir
        self.relative_image_dir = Path(image_dir)  # Для сравнения с записями в labels.json

        self.root = tk.Tk()
        self.root.title("Image Labeling Tool")
        self.root.geometry("1000x900")

        self.categories = categories
        self.current_image_path = None
        self.labels = {}
        self.dropdown_after_id = None
        self.history = []

        # Путь для сохранения результатов (labels.json)
        self.labels_file = self.image_dir / "labels.json"
        if self.labels_file.exists():
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                self.labels = json.load(f)

        self.appellations = self.fetch_appellations()
        self.descriptions = self.fetch_descriptions()

        self.select_mode()

        # Формируем список изображений
        self.image_paths = []
        for product_dir in self.image_dir.iterdir():
            if product_dir.is_dir():
                for img_path in product_dir.glob("*.jpg"):
                    # Формируем относительный путь в том же формате, что в labels.json
                    rel_path = str(self.relative_image_dir / img_path.relative_to(self.image_dir))
                    rel_path = rel_path.replace('/', '\\')  # Приводим слеши к единому формату
                    if self.mode.get() == "unlabeled" and rel_path in self.labels:
                        continue
                    self.image_paths.append(img_path)
        self.current_index = 0

        self.setup_gui()
        self.load_image()

    def fetch_appellations(self):
        """Получает названия товаров по API."""
        url = "http://api.athebyme-market.ru:8081/api/appellations"
        try:
            response = requests.post(url, json={"productIDs": []})
            response.raise_for_status()
            data = response.json()
            if "productIDs" in data:
                return data["productIDs"]
            return data
        except Exception as e:
            print("Ошибка получения апелляций:", e)
            return {}

    def fetch_descriptions(self):
        """Получает описания товаров по API."""
        url = "http://api.athebyme-market.ru:8081/api/descriptions"
        try:
            response = requests.post(url, json={"productIDs": []})
            response.raise_for_status()
            data = response.json()
            if "productIDs" in data:
                return data["productIDs"]
            return data
        except Exception as e:
            print("Ошибка получения описаний:", e)
            return {}

    def select_mode(self):
        """Показываем диалог выбора режима перед открытием основного окна."""
        # Скрываем основное окно, чтобы диалог точно был на переднем плане
        self.root.withdraw()
        self.mode = tk.StringVar(value="unlabeled")
        dialog = tk.Toplevel()
        dialog.title("Выберите режим разметки")
        dialog.attributes("-topmost", True)  # окно поверх всех остальных
        dialog.geometry("400x200+500+300")
        dialog.grab_set()  # делаем модальным

        label = ttk.Label(dialog, text="Выберите режим разметки:")
        label.pack(pady=10, padx=20)

        rb1 = ttk.Radiobutton(
            dialog,
            text="Начать разметку сначала (все изображения)",
            variable=self.mode,
            value="all"
        )
        rb1.pack(anchor=tk.W, padx=20, pady=5)

        rb2 = ttk.Radiobutton(
            dialog,
            text="Использовать только необработанные изображения",
            variable=self.mode,
            value="unlabeled"
        )
        rb2.pack(anchor=tk.W, padx=20, pady=5)

        ok_button = ttk.Button(dialog, text="OK", command=dialog.destroy)
        ok_button.pack(pady=10)

        self.root.wait_window(dialog)
        self.root.deiconify()

    def setup_gui(self):
        self.display_frame = ttk.Frame(self.root)
        self.display_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Колонки, чтобы изображение и информация располагались рядом
        self.display_frame.columnconfigure(0, weight=0)
        self.display_frame.columnconfigure(1, weight=1)

        # Метка для изображения
        self.image_label = tk.Label(self.display_frame)
        self.image_label.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="nsew")

        # Фрейм для информации (в нём – короткая часть в Label и длинное описание в ScrolledText)
        self.info_frame = ttk.Frame(self.display_frame)
        self.info_frame.grid(row=0, column=1, padx=(5, 10), pady=5, sticky="nw")

        # Короткая часть (ID и Название) – в Label
        self.product_title_label = ttk.Label(
            self.info_frame,
            text="",
            wraplength=250,
            justify="left",
            font=("Arial", 12)
        )
        self.product_title_label.pack(anchor="nw", fill="x", pady=(0, 5))

        # Длинное описание – в прокручиваемом тексте
        self.desc_text = ScrolledText(
            self.info_frame,
            wrap="word",
            width=40,     # ограничиваем ширину
            height=10,    # ограничиваем высоту (10 строк)
            font=("Arial", 10)
        )
        self.desc_text.pack(anchor="nw", fill="both", expand=False)
        # Панель с прокруткой для кнопок категорий
        category_panel_container = ttk.Frame(self.root)
        category_panel_container.pack(pady=10, fill=tk.X, padx=10)

        self.category_canvas = tk.Canvas(category_panel_container, height=200)
        scrollbar = ttk.Scrollbar(category_panel_container, orient="vertical", command=self.category_canvas.yview)
        self.category_canvas.configure(yscrollcommand=scrollbar.set)

        self.category_frame = ttk.Frame(self.category_canvas)
        self.category_frame.bind(
            "<Configure>",
            lambda e: self.category_canvas.configure(scrollregion=self.category_canvas.bbox("all"))
        )
        self.category_canvas.create_window((0, 0), window=self.category_frame, anchor="nw")
        self.category_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Привязка прокрутки мыши к Canvas
        self.root.bind_all("<MouseWheel>", self.on_mousewheel)
        self.root.bind_all("<Button-4>", self.on_mousewheel)
        self.root.bind_all("<Button-5>", self.on_mousewheel)

        # Размещаем кнопки категорий
        num_cols = 5
        for index, category in enumerate(self.categories):
            btn = ttk.Button(
                self.category_frame,
                text=category,
                command=lambda c=category: self.label_image(c),
                width=25
            )
            row = index // num_cols
            col = index % num_cols
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="w")

        # Фрейм для навигации и кнопок "Back" и "Skip"
        self.nav_frame = ttk.Frame(self.root)
        self.nav_frame.pack(pady=5)

        self.back_btn = ttk.Button(self.nav_frame, text="Back", command=self.go_back)
        self.back_btn.pack(side=tk.LEFT, padx=5)
        self.skip_btn = ttk.Button(self.nav_frame, text="Skip", command=self.load_next_image)
        self.skip_btn.pack(side=tk.LEFT, padx=5)

        self.apply_to_all_var = tk.BooleanVar(value=True)
        self.apply_to_all_checkbox = ttk.Checkbutton(
            self.nav_frame,
            text="Применить для всех фото товара",
            variable=self.apply_to_all_var
        )
        self.apply_to_all_checkbox.pack(side=tk.LEFT, padx=5)

        # Поиск по категории (Combobox)
        self.search_frame = ttk.Frame(self.root)
        self.search_frame.pack(pady=10)

        search_label = ttk.Label(self.search_frame, text="Поиск по категории:")
        search_label.pack(side=tk.LEFT, padx=5)

        self.category_var = tk.StringVar()
        # state="normal" позволяет печатать
        self.category_combobox = ttk.Combobox(
            self.search_frame,
            textvariable=self.category_var,
            values=self.categories,
            width=40,
            state="normal"
        )
        self.category_combobox.pack(side=tk.LEFT, padx=5)
        self.category_combobox.bind("<KeyRelease>", self.on_combobox_keyrelease)

        self.search_btn = ttk.Button(
            self.search_frame,
            text="Применить",
            command=self.label_image_from_combobox
        )
        self.search_btn.pack(side=tk.LEFT, padx=5)

        # Прогресс и кнопка сохранения
        self.progress_label = ttk.Label(self.root, text="")
        self.progress_label.pack(pady=5)

        self.save_btn = ttk.Button(self.root, text="Save and Exit", command=self.save_and_exit)
        self.save_btn.pack(pady=5)

        self.current_photo = None  # Добавляем для хранения текущего PhotoImage
        self.max_history_size = 100  # Ограничиваем размер истории

        # Создаем слабый словарь для кэширования изображений
        self.image_cache = {}

    def on_mousewheel(self, event):
        """Обработчик события прокрутки колеса мыши"""
        try:
            # Получаем координаты курсора
            x, y = self.root.winfo_pointerxy()

            try:
                widget_under_cursor = self.root.winfo_containing(x, y)
            except (KeyError, TypeError):
                # Если не удалось определить виджет (например, выпадающий список)
                return "break"

            if widget_under_cursor is None:
                return "break"

            # Проверяем, находится ли курсор над canvas или его содержимым
            canvas = self.category_canvas
            if widget_under_cursor == canvas or \
                    str(canvas) in str(widget_under_cursor) or \
                    widget_under_cursor.winfo_parent() == str(canvas):

                # Определяем направление прокрутки
                if hasattr(event, 'delta'):
                    delta = int(-1 * (event.delta / 120))
                elif event.num == 4:
                    delta = -1
                elif event.num == 5:
                    delta = 1
                else:
                    return "break"

                # Проверяем, есть ли что прокручивать
                bbox = canvas.bbox("all")
                if bbox:
                    canvas.yview_scroll(delta, "units")

            return "break"
        except Exception as e:
            print(f"Error in mousewheel handler: {e}")
            return "break"

    def on_combobox_keyrelease(self, event):
        """Фильтруем список категорий с дебаунсом, чтобы не прерывать ввод."""
        typed_text = self.category_var.get().strip()
        typed_lower = typed_text.lower()

        if typed_lower == '':
            data = self.categories
        else:
            data = [cat for cat in self.categories if typed_lower in cat.lower()]
        self.category_combobox['values'] = data

        if self.dropdown_after_id is not None:
            self.root.after_cancel(self.dropdown_after_id)
        self.dropdown_after_id = self.root.after(500, self.open_dropdown)

    def open_dropdown(self):
        """Открываем список и возвращаем фокус в поле ввода."""
        self.category_combobox.event_generate("<Down>")
        self.category_combobox.focus()
        self.category_combobox.icursor(tk.END)
        self.dropdown_after_id = None

    def label_image_from_combobox(self):
        """Применяем выбранную метку, если она корректна."""
        category = self.category_var.get().strip()
        if category in self.categories:
            self.label_image(category)
        else:
            print("Выберите корректную категорию из списка.")

    def load_image(self):
        if self.current_index < 0:
            self.current_index = 0
        if self.current_index >= len(self.image_paths):
            self.progress_label.config(text="All images have been labeled!")
            self.image_label.config(image="")
            self.product_title_label.config(text="")
            self.desc_text.delete("1.0", tk.END)
            return

        # Очищаем предыдущее изображение
        if self.current_photo:
            self.image_label.config(image="")
            self.current_photo = None

        self.current_image_path = self.image_paths[self.current_index]
        self.progress_label.config(text=f"Progress: {self.current_index + 1}/{len(self.image_paths)}")

        try:
            # Попытка загрузить изображение
            image = Image.open(self.current_image_path)
            display_size = (600, 400)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)

            self.current_photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.current_photo)

            # Явно закрываем файл изображения
            image.close()

        except Exception as e:
            print(f"Error loading image {self.current_image_path}: {e}")
            self.current_photo = None
            self.image_label.config(image="")

        # Обновляем информацию о продукте
        product_id = self.current_image_path.parent.name
        product_name = self.appellations.get(product_id, "Неизвестный товар")
        product_desc = self.descriptions.get(product_id, "Нет описания")

        self.product_title_label.config(
            text=f"Product ID: {product_id}\nНазвание: {product_name}"
        )
        self.desc_text.delete("1.0", tk.END)
        self.desc_text.insert(tk.END, f"Описание: {product_desc}")

        self.root.title(f"Image Labeler - {self.current_image_path}")

        gc.collect()

    def load_next_image(self):
        self.current_index += 1
        self.load_image()

    def go_back(self):
        """Отмена последнего действия."""
        if not self.history:
            return

        record = self.history.pop()
        if record['type'] == 'single':
            image_str = record['image']
            if image_str in self.labels:
                del self.labels[image_str]
            self.current_index = record['index']
        elif record['type'] == 'product':
            # Удаляем метки
            for img_str in record['images']:
                if img_str in self.labels:
                    del self.labels[img_str]

            # Преобразуем относительные пути в абсолютные при восстановлении
            images = []
            for img_str in record['images']:
                # Конвертируем Windows-пути в системные
                img_str = img_str.replace('\\', '/')
                # Создаем абсолютный путь
                abs_path = self.image_dir / Path(img_str).relative_to(self.relative_image_dir)
                if abs_path.exists():  # Проверяем существование файла
                    images.append(abs_path)

            # Восстанавливаем изображения в список
            if images:  # Проверяем, что есть что восстанавливать
                self.image_paths[record['index']:record['index']] = images
                self.current_index = record['index']
            else:
                print("Warning: No valid images to restore")
                # Если нет валидных изображений, просто двигаемся к следующему
                self.current_index = min(record['index'], len(self.image_paths) - 1)

        self.save_labels()
        self.load_image()

    def save_labels(self):
        with open(self.labels_file, 'w', encoding="utf-8") as f:
            json.dump(self.labels, f, indent=2, ensure_ascii=False)

    def label_image(self, category):
        if not self.current_image_path:
            return

        # Создаем относительный путь для ключа в labels
        try:
            rel_path = self.current_image_path.relative_to(self.image_dir)
            key = str(self.relative_image_dir / rel_path)
            key = key.replace('/', '\\')  # Приводим к Windows-формату для совместимости
        except ValueError as e:
            print(f"Error creating relative path: {e}")
            return

        if self.apply_to_all_var.get():
            current_product = self.current_image_path.parent
            images_to_label = [p for p in self.image_paths if p.parent == current_product]

            # Создаем список относительных путей для истории
            relative_paths = []
            for p in images_to_label:
                try:
                    rel_p = p.relative_to(self.image_dir)
                    rel_path_str = str(self.relative_image_dir / rel_p).replace('/', '\\')
                    relative_paths.append(rel_path_str)
                except ValueError as e:
                    print(f"Error creating relative path for {p}: {e}")
                    continue

            self.history.append({
                'type': 'product',
                'product': str(current_product),
                'images': relative_paths,
                'index': self.current_index
            })

            # Добавляем метки
            for p in images_to_label:
                try:
                    rel_p = p.relative_to(self.image_dir)
                    k = str(self.relative_image_dir / rel_p).replace('/', '\\')
                    self.labels[k] = category
                except ValueError as e:
                    print(f"Error labeling image {p}: {e}")
                    continue

            # Удаляем обработанные изображения из списка
            self.image_paths = [p for p in self.image_paths if p.parent != current_product]
            if self.current_index >= len(self.image_paths):
                self.current_index = len(self.image_paths) - 1
        else:
            self.history.append({
                'type': 'single',
                'image': key,
                'index': self.current_index
            })
            self.labels[key] = category
            self.current_index += 1

        self.save_labels()
        self.load_image()

    def save_and_exit(self):
        self.save_labels()
        if self.current_photo:
            self.image_label.config(image="")
            self.current_photo = None
        self.root.quit()

    def run(self):
        self.root.mainloop()


def main():
    categories = [
        "Аксессуары для игрушек эротик",
        "Анальные бусы",
        "Анальные груши",
        "Анальные крюки",
        "Анальные пробки",
        "Анальные шарики",
        "Ароматизаторы эротик",
        "Бандажи эротик",
        "Благовония эротик",
        "Боди эротик",
        "Браслеты эротик",
        "Бюстгальтеры эротик",
        "Вагинальные тренажеры",
        "Вагинальные шарики",
        "Вакуумно-волновые стимуляторы",
        "Вакуумные помпы эротик",
        "Веревки для бондажа",
        "Вибраторы",
        "Вибропули",
        "Вибротрусики",
        "Виброяйца",
        "Возбуждающие напитки",
        "Возбуждающие препараты",
        "Возбуждающие средства",
        "Галстуки эротик",
        "Гартеры эротик",
        "Гидропомпы эротик",
        "Головные уборы эротик",
        "Гольфы эротик",
        "Грации эротик",
        "Дезодоранты эротик",
        "Зажимы для сосков",
        "Запчасти для секс машины",
        "Имитаторы груди",
        "Календари эротические",
        "Кляпы эротик",
        "Колготки эротик",
        "Колеса Вартенберга",
        "Комбинезоны эротик",
        "Комплекты БДСМ",
        "Комплекты эротик",
        "Концентраты феромонов",
        "Корсеты эротик",
        "Леггинсы эротик",
        "Леденцы эротик",
        "Ленты эротик",
        "Лубриканты",
        "Манжеты эротик",
        "Мармелад эротик",
        "Маски эротик",
        "Массажеры простаты",
        "Массажные средства эротик",
        "Мастурбаторы мужские",
        "Мыло эротик",
        "Наборы игрушек для взрослых",
        "Накидки эротик",
        "Наручники эротик",
        "Насадки для вибраторов",
        "Насадки на мастурбатор",
        "Насадки на страпон",
        "Насадки на член",
        "Насосы для секс кукол",
        "Насосы на член",
        "Настольные игры для взрослых",
        "Настольные игры эротик",
        "Неглиже эротик",
        "Оковы эротик",
        "Ошейники эротик",
        "Пеньюары эротик",
        "Перчатки эротик",
        "Печенье эротик",
        "Платья эротик",
        "Плетки эротик",
        "Поводки эротик",
        "Подвязки эротик",
        "Подушки эротик",
        "Помады эротик",
        "Портупеи эротик",
        "Посуда эротик",
        "Пояса верности",
        "Пояса эротик",
        "Приспособления для интимной косметики",
        "Пролонгаторы",
        "Презервативы",
        "Простыни БДСМ",
        "Пульсаторы",
        "Пэстис эротик",
        "Расширители гинекологические",
        "Ролевые костюмы эротик",
        "Руки для фистинга",
        "Свечи БДСМ",
        "Свечи эротик",
        "Секс качели",
        "Секс кресла",
        "Секс куклы",
        "Секс машины",
        "Секс мячи",
        "Слепки эротик",
        "Спринцовки эротик",
        "Средства для очистки секс-игрушек",
        "Средства с феромонами",
        "Стопы для фистинга",
        "Страпоны",
        "Стрэпы эротик",
        "Стэки эротик",
        "Сувениры эротик",
        "Сумки для секс машины",
        "Топы эротик",
        "Трусы для страпона",
        "Трусы эротик",
        "Увеличители члена",
        "Указки эротик",
        "Уретральные зонды",
        "Утяжелители эротик",
        "Уходовые средства эротик",
        "Фаллоимитаторы",
        "Фаллопротезы",
        "Фиксаторы эротик",
        "Чокеры эротик",
        "Чулки эротик",
        "Шлепалки эротик",
        "Шорты эротик",
        "Щеточки эротик",
        "Электростимуляторы",
        "Эрекционные кольца",
        "Юбки эротик",
        "Другое"
    ]
    image_dir = "dataset/raw_images"
    labeler = ImageLabeler(image_dir, categories)
    labeler.run()


if __name__ == "__main__":
    main()
