# main.py
import os
import json
import builtins

# --- Charger le fichier config.json ---
CONFIG_PATH = "./config/config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

# --- Filtrer les prints DEBUG si besoin ---
if not config.get("debug", False):
    old_print = print
    def print(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith("[DBG]"):
            return
        old_print(*args, **kwargs)


from datetime import datetime
from src.align_and_compare import process_one

INPUT_DIR = "./bulletins-a-trier"

def main():
    # 1. créer le dossier export avec timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_root = os.path.join("export", ts)
    os.makedirs(export_root, exist_ok=True)

    for name in os.listdir(INPUT_DIR):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue
        path = os.path.join(INPUT_DIR, name)
        process_one(path, export_root=export_root)

if __name__ == "__main__":
    main()
