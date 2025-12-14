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

def parse_mode_list(text, min_v, max_v):
    """
    Разбирает ввод для выбора режимов/комбинаций эффектов.
    Поддерживает:
      - 'all' -> [1,2,...,max_v]
      - '3' -> [3]
      - '2-5' -> [2,3,4,5]
      - '9+12' -> [(9,12)]  (кортеж означает последовательное применение 9 затем 12)
      - '9+12,3,5-6' -> [(9,12), 3, 5, 6]

    Возвращает список элементов, где элемент — int (один эффект) или tuple(int,...) (последовательность).
    Возвращает [] при некорректном вводе.
    """
    text = (text or "").strip().lower()
    if not text:
        return []

    if text == "all":
        return list(range(min_v, max_v + 1))

    out = []
    # разделяем по запятым (ввод может быть '9+12,3,5-6')
    parts = [p.strip() for p in text.split(",") if p.strip()]
    for part in parts:
        # каждый part может быть 'a+b+c' или 'n' или 'n-m'
        if "+" in part:
            sub = []
            ok = True
            for token in part.split("+"):
                token = token.strip()
                if not token:
                    ok = False; break
                # поддержим диапазоны внутри +, например "2-4+7" -> 2,3,4,7
                if "-" in token:
                    try:
                        a, b = map(int, token.split("-", 1))
                        for v in range(a, b+1):
                            if min_v <= v <= max_v:
                                sub.append(int(v))
                    except:
                        ok = False
                        break
                else:
                    try:
                        v = int(token)
                        if min_v <= v <= max_v:
                            sub.append(int(v))
                        else:
                            ok = False; break
                    except:
                        ok = False; break
            if ok and sub:
                out.append(tuple(sub))
        else:
            # одиночный токен или диапазон
            token = part
            if "-" in token:
                try:
                    a, b = map(int, token.split("-", 1))
                    for v in range(a, b+1):
                        if min_v <= v <= max_v:
                            out.append(int(v))
                except:
                    continue
            else:
                try:
                    v = int(token)
                    if min_v <= v <= max_v:
                        out.append(int(v))
                except:
                    continue
    return out