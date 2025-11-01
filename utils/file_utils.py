# utils/file_utils.py
import os

def resolve_shortcut(path):
    if not path.lower().endswith(".lnk"):
        return path
    try:
        import win32com.client
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(path)
        target = shortcut.Targetpath
        if target and os.path.exists(target):
            print(f"Ярлык → {target}")
            return target
    except Exception as e:
        print(f"[!] Ярлык: {e}")
    return path