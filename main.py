# main.py
import os
import json
import logging
from datetime import datetime
from src.align_and_compare import process_one

# --- Charger le fichier config.json ---
CONFIG_PATH = "./config/config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

# --- Configuration du logging ---
log_level = config.get("log_level", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level.upper(), logging.INFO),
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

INPUT_DIR = "./bulletins-a-trier"

def main():
    # 1. créer le dossier export avec timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_root = os.path.join("export", ts)
    os.makedirs(export_root, exist_ok=True)
    
    logger.info(f"Démarrage du traitement - export vers {export_root}")

    for name in os.listdir(INPUT_DIR):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue
        path = os.path.join(INPUT_DIR, name)
        process_one(path, export_root=export_root)
    
    logger.info("Traitement terminé")

if __name__ == "__main__":
    main()