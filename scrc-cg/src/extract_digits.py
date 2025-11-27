#!/usr/bin/env python3
# src/extract_digits.py
"""
Script d'extraction des chiffres depuis les bulletins de référence.

Extrait les digits de la zone d'identification de chaque bulletin
et les sauvegarde dans ref/digits_extracted/ pour permettre de
choisir manuellement les meilleurs templates.

Usage:
    python src/extract_digits.py
    python src/extract_digits.py --debug
    python src/extract_digits.py --output ./mon_dossier
"""

import os
import json
import cv2
import logging
import numpy as np
import argparse
from pathlib import Path

# Chemins de base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../src
CONFIG_PATH = os.path.join(BASE_DIR, "..", "config", "config.json")
REF_BULLETINS_DIR = os.path.join(BASE_DIR, "..", "ref", "bulletins")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "..", "ref", "digits_extracted")

logger = logging.getLogger(__name__)


def load_config():
    """Charge la configuration depuis config.json."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def crop_identification_zone(img, config):
    """
    Extrait la zone d'identification définie dans config.json.
    
    Adapte automatiquement les coordonnées si l'image a une résolution
    différente de la résolution de référence (2480x3500).
    """
    z = config["zones"]["identification"]
    x, y, w, h = int(z["x"]), int(z["y"]), int(z["w"]), int(z["h"])
    
    # Résolution de référence (pour laquelle la config a été créée)
    REF_WIDTH = 2480
    REF_HEIGHT = 3500
    
    # Résolution actuelle de l'image
    img_h, img_w = img.shape[:2]
    
    # Calculer le ratio si nécessaire
    if abs(img_w - REF_WIDTH) > 100 or abs(img_h - REF_HEIGHT) > 100:
        ratio_w = img_w / REF_WIDTH
        ratio_h = img_h / REF_HEIGHT
        
        x = int(x * ratio_w)
        y = int(y * ratio_h)
        w = int(w * ratio_w)
        h = int(h * ratio_h)
        
        logger.debug(f"Adaptation résolution: ratio={ratio_w:.3f}x{ratio_h:.3f}, zone=({x},{y},{w},{h})")
    
    # Vérifier les limites
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    
    return img[y:y+h, x:x+w].copy()


def binarize(img):
    """Convertit une image en binaire (noir et blanc)."""
    # Gérer les images déjà en niveaux de gris
    if len(img.shape) == 2:
        gray = img
    elif len(img.shape) == 3 and img.shape[2] == 1:
        gray = img[:, :, 0]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Amélioration du contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Binarisation Otsu
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def crop_to_content(binary_img, pad=2):
    """
    Recadre une image binaire autour du contenu (pixels noirs).
    Ajoute un padding blanc autour.
    """
    ys, xs = np.where(binary_img == 0)
    if len(xs) == 0 or len(ys) == 0:
        return binary_img
    
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    
    # Appliquer le padding
    x_min = max(x_min - pad, 0)
    y_min = max(y_min - pad, 0)
    x_max = min(x_max + pad, binary_img.shape[1] - 1)
    y_max = min(y_max + pad, binary_img.shape[0] - 1)
    
    return binary_img[y_min:y_max+1, x_min:x_max+1]


def segment_digits(binary_zone, debug=False):
    """
    Segmente les chiffres d'une zone binaire.
    
    Returns:
        Liste de tuples (x_position, image_digit) triés de gauche à droite
    """
    # Nettoyage morphologique
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(binary_zone, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Inverser pour trouver les contours (objets blancs sur fond noir)
    inv = 255 - clean
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digits = []
    boxes = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filtrer le bruit (trop petit)
        if w < 10 or h < 20:
            continue
        
        # Filtrer les lignes horizontales (trait SNL par exemple)
        if w > h * 3:
            continue
        
        roi = binary_zone[y:y+h, x:x+w]
        digits.append((x, roi))
        boxes.append((x, y, w, h))
    
    # Trier de gauche à droite
    digits.sort(key=lambda t: t[0])
    
    if debug:
        dbg = cv2.cvtColor(binary_zone, cv2.COLOR_GRAY2BGR)
        for i, (x, y, w, h) in enumerate(sorted(boxes, key=lambda b: b[0])):
            cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(dbg, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Segmentation", dbg)
        cv2.waitKey(0)
    
    return digits


def extract_list_number_from_filename(filename):
    """
    Extrait le numéro de liste depuis le nom du fichier.
    Ex: "Bulletin_LED_001.png" -> "01"
        "Bulletin_VERTS_002.png" -> "02"
        "Bulletin_SNL_000.png" -> "SNL" (cas spécial)
    """
    # Chercher le pattern _XXX. où XXX sont des chiffres
    import re
    match = re.search(r'_(\d{3})\.', filename)
    if match:
        num = match.group(1)
        # Convertir "001" -> "01", "002" -> "02", etc.
        if num == "000":
            return "SNL"
        return num.lstrip("0").zfill(2) if num != "000" else "00"
    return None


def process_bulletin(image_path, config, output_dir, debug=False):
    """
    Traite un bulletin et extrait ses digits.
    
    Returns:
        Liste des digits extraits avec leurs métadonnées
    """
    filename = os.path.basename(image_path)
    logger.info(f"Traitement de {filename}")
    
    # Charger l'image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Impossible de lire {image_path}")
        return []
    
    # Extraire le numéro de liste attendu
    expected_number = extract_list_number_from_filename(filename)
    if expected_number == "SNL":
        logger.info(f"  → Bulletin SNL ignoré (pas de numéro)")
        return []
    
    logger.info(f"  → Numéro attendu: {expected_number}")
    
    # Extraire la zone d'identification
    zone = crop_identification_zone(img, config)
    bin_zone = binarize(zone)
    
    if debug:
        cv2.imshow(f"Zone - {filename}", zone)
        cv2.imshow(f"Binaire - {filename}", bin_zone)
    
    # Segmenter les digits
    digit_segments = segment_digits(bin_zone, debug=debug)
    
    if not digit_segments:
        logger.warning(f"  → Aucun digit trouvé!")
        return []
    
    logger.info(f"  → {len(digit_segments)} digit(s) trouvé(s)")
    
    # Extraire et sauvegarder chaque digit
    extracted = []
    expected_digits = list(expected_number)  # "01" -> ["0", "1"]
    
    for i, (x_pos, digit_img) in enumerate(digit_segments):
        # Recadrer autour du contenu
        digit_cropped = crop_to_content(digit_img, pad=4)
        
        # Déterminer le chiffre attendu
        if i < len(expected_digits):
            digit_value = expected_digits[i]
        else:
            digit_value = "unknown"
        
        extracted.append({
            "value": digit_value,
            "image": digit_cropped,
            "source": filename,
            "position": i
        })
        
        if debug:
            cv2.imshow(f"Digit {i} ({digit_value})", digit_cropped)
    
    if debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return extracted


def save_digits(digits_by_value, output_dir):
    """
    Sauvegarde les digits extraits dans le dossier de sortie.
    
    Naming convention:
    - Premier exemplaire: 0.png
    - Suivants: 0(1).png, 0(2).png, etc.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for digit_value, digit_list in digits_by_value.items():
        for i, digit_info in enumerate(digit_list):
            if i == 0:
                filename = f"{digit_value}.png"
            else:
                filename = f"{digit_value}({i}).png"
            
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, digit_info["image"])
            logger.info(f"  Sauvegardé: {filename} (source: {digit_info['source']})")


def main():
    parser = argparse.ArgumentParser(
        description="Extrait les digits des bulletins de référence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
    python src/extract_digits.py
    python src/extract_digits.py --debug
    python src/extract_digits.py --output ./mes_digits
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Dossier de sortie (défaut: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Afficher les étapes intermédiaires"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Afficher plus de détails"
    )
    
    args = parser.parse_args()
    
    # Configuration du logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(message)s"
    )
    
    # Charger la config
    config = load_config()
    
    # Trouver les bulletins de référence
    ref_dir = config["paths"].get("ref_bulletins_dir", REF_BULLETINS_DIR)
    if not os.path.isabs(ref_dir):
        ref_dir = os.path.join(BASE_DIR, "..", ref_dir)
    
    if not os.path.exists(ref_dir):
        logger.error(f"Dossier de référence introuvable: {ref_dir}")
        return 1
    
    # Lister les images
    image_files = []
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        image_files.extend(Path(ref_dir).glob(f"*{ext}"))
        image_files.extend(Path(ref_dir).glob(f"*{ext.upper()}"))
    
    image_files = sorted(set(image_files))
    
    if not image_files:
        logger.error(f"Aucune image trouvée dans {ref_dir}")
        return 1
    
    print()
    print("=" * 60)
    print("EXTRACTION DES DIGITS")
    print("=" * 60)
    print(f"Source      : {ref_dir}")
    print(f"Destination : {args.output}")
    print(f"Images      : {len(image_files)}")
    print("=" * 60)
    print()
    
    # Collecter tous les digits
    digits_by_value = {}
    
    for img_path in image_files:
        extracted = process_bulletin(str(img_path), config, args.output, debug=args.debug)
        
        for digit_info in extracted:
            value = digit_info["value"]
            if value not in digits_by_value:
                digits_by_value[value] = []
            digits_by_value[value].append(digit_info)
    
    # Résumé
    print()
    print("=" * 60)
    print("RÉSUMÉ DES DIGITS EXTRAITS")
    print("=" * 60)
    
    total = 0
    for value in sorted(digits_by_value.keys()):
        count = len(digits_by_value[value])
        total += count
        sources = [d["source"] for d in digits_by_value[value]]
        print(f"  Digit '{value}': {count} exemplaire(s)")
        for src in sources:
            print(f"           └─ {src}")
    
    print(f"\n  Total: {total} digit(s)")
    print("=" * 60)
    print()
    
    # Sauvegarder
    print("Sauvegarde des digits...")
    save_digits(digits_by_value, args.output)
    
    print()
    print(f"✓ Extraction terminée! Vérifiez les résultats dans: {args.output}")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())