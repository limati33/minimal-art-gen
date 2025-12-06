# utils/input_utils.py
from .logging_utils import YELLOW, RED, GREEN, RESET

def print_progress(step, total_steps=5, prefix=""):
    percent = step / total_steps * 100
    bar_len = 20
    filled = int(bar_len * step // total_steps)
    bar = f"{GREEN}{'█' * filled}{RESET}{'.' * (bar_len - filled)}"
    print(f"\r{prefix}{bar} {YELLOW}{percent:3.0f}%{RESET}", end="", flush=True)

def ask_int(prompt, min_v, max_v):
    while True:
        try:
            v = int(input(f"{YELLOW}{prompt}{RESET}"))
            if min_v <= v <= max_v: return v
            print(f"{RED}От {min_v} до {max_v}.{RESET}")
        except: print(f"{RED}Целое число!{RESET}")

def ask_float(prompt, min_v, max_v):
    while True:
        try:
            v = float(input(f"{YELLOW}{prompt}{RESET}"))
            if min_v <= v <= max_v: return v
            print(f"{RED}От {min_v} до {max_v}.{RESET}")
        except: print(f"{RED}Число!{RESET}")

def parse_int_list(text, min_v, max_v):
    """
    Разбирает ввод пользователя:
    '3,5,7' -> [3,5,7]
    '2-5' -> [2,3,4,5]
    'all' -> полный диапазон
    'no limit' -> None (без ограничения)
    """
    text = text.strip().lower()

    if text in ("no limit", "nolimit", "∞", "inf", "unlimited"):
        return None

    if text == "all":
        return list(range(min_v, max_v + 1))

    result = set()
    for part in text.replace(" ", "").split(","):
        if "-" in part:
            try:
                start, end = map(int, part.split("-"))
                for i in range(start, end + 1):
                    if min_v <= i <= max_v:
                        result.add(i)
            except:
                pass
        else:
            try:
                val = int(part)
                if min_v <= val <= max_v:
                    result.add(val)
            except:
                pass
    return sorted(result)
