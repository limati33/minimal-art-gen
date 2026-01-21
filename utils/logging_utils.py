# utils/logging_utils.py
import os
import datetime
import traceback
from colorama import init, Fore, Style

init(autoreset=True)

# Цвета
RESET = Style.RESET_ALL
BOLD = Style.BRIGHT
CYAN = Fore.CYAN
GREEN = Fore.GREEN
YELLOW = Fore.YELLOW
RED = Fore.RED
MAGENTA = Fore.MAGENTA
BLUE = Fore.BLUE

# === Пути ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_minimal_art")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Названия эффектов ===
EFFECT_NAMES = {
    1: "Постеризация", 2: "Плавные пятна", 3: "Комикс", 4: "Чернила", 5: "Сетка",
    6: "Пыль", 7: "Мел", 8: "ASCII-Арт", 9: "Пластик", 10: "Ресто LCD",
    11: "Векторизация", 12: "Свеча", 13: "Голограмма", 14: "Акварель", 15: "Технический",
    16: "Карта", 17: "Мозаика", 18: "Сон", 19: "Пиксель-Арт", 20: "Огненный",
    21: "Рентген", 22: "Графика", 23: "Vice City", 24: "TV box", 25: "Винтаж",
    26: "Плёнка", 27: "Принтер", 28: "Штрих-тень", 29: "Цвето-фильтер", 30: "Неон",
    31: "Дрифт", 32: "Под водой"
}

# === Логирование ошибок ===
def log_error(message: str, exc=None, image_path=None, n_colors=None, blur=None, mode=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(LOG_DIR, f"error_{timestamp}.log")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== ОШИБКА ===\n")
        f.write(f"Время: {datetime.datetime.now()}\n")
        if image_path: f.write(f"Файл: {image_path}\n")
        if n_colors is not None: f.write(f"Цветов: {n_colors}\n")
        if blur is not None: f.write(f"Размытие: {blur}\n")
        if mode is not None: f.write(f"Эффект: {mode} ({EFFECT_NAMES.get(mode, 'Неизвестно')})\n")
        f.write(f"\nСообщение:\n{message}\n")
        if exc:
            f.write(f"\nТип: {type(exc).__name__}\n")
            f.write(f"Аргументы: {exc.args}\n")
            f.write(f"\nTraceback:\n{traceback.format_exc()}")

    print(f"{RED}\n[ОШИБКА] Лог: {log_path}{RESET}")
    return log_path
