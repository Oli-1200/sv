# src/align_and_compare.py
"""
Module de comparaison d'images de bulletins de vote.

AMÉLIORATIONS v2:
- Conversion N&B et redimensionnement intégrés
- Nouvelle méthode de comparaison "structurelle" qui détecte les ajouts
  de texte plutôt qu'une simple différence pixel-à-pixel
- Meilleure tolérance au bruit de scan et aux petits décalages
- Conservation de l'image originale pour la copie finale
"""

import os
import csv
import cv2
import json
import logging
import numpy as np
from shutil import copyfile

from src.detect_corners import detect_corners
from src.identify_list_number import identify_list_number
from src.convert_bw import convert_and_resize_for_comparison, convert_for_comparison

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "..", "config", "config.json")

logger = logging.getLogger(__name__)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def append_report(export_root, row):
    report_path = os.path.join(export_root, "report.csv")
    os.makedirs(export_root, exist_ok=True)
    file_exists = os.path.exists(report_path)
    try:
        with open(report_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            if not file_exists:
                writer.writerow([
                    "fichier",
                    "numero_liste",
                    "template",
                    "zones_ajoutees",
                    "surface_ajoutee",
                    "pixels_diff_soft",
                    "pixels_diff_hard",
                    "statut"
                ])
            writer.writerow(row)
    except PermissionError:
        alt_path = os.path.join(export_root, "report_backup.csv")
        logger.warning(f"report.csv verrouillé, écriture dans {alt_path}")
        with open(alt_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(row)


def apply_exclusions(img, config):
    """Applique les zones d'exclusion (masque en blanc)."""
    ex = config["zones"].get("exclusion", [])
    for zone in ex:
        x, y, w, h = [int(zone[k]) for k in ("x", "y", "w", "h")]
        img[y:y+h, x:x+w] = 255
    return img


def align_images(src_img, ref_img, src_corners, ref_corners):
    """Aligne l'image source sur la référence via transformation perspective."""
    def order(cdict):
        return np.array([
            cdict["top_left"],
            cdict["top_right"],
            cdict["bottom_left"],
            cdict["bottom_right"]
        ], dtype=np.float32)

    pts_src = order(src_corners)
    pts_ref = order(ref_corners)
    M = cv2.getPerspectiveTransform(pts_src, pts_ref)
    aligned = cv2.warpPerspective(src_img, M, (ref_img.shape[1], ref_img.shape[0]))
    return aligned


def compare_images_structural(aligned, ref_img, config, debug_path=None, is_snl=False):
    """
    NOUVELLE MÉTHODE: Comparaison structurelle.
    
    Détecte les zones de TEXTE AJOUTÉ plutôt qu'une simple différence
    pixel-à-pixel. Beaucoup plus robuste au bruit de scan.
    
    Principe:
    1. Inverser les images (texte en blanc sur fond noir)
    2. Dilater pour épaissir le texte et combler les petits trous
    3. Soustraire la référence de l'image alignée
    4. Ce qui reste = texte ajouté (modifications manuscrites)
    5. Filtrer les petits artefacts
    
    Returns:
        status: "modifie" ou "pas-modifie"
        counts: dict avec statistiques détaillées
    """
    diff_conf = config.get("diff", {})
    
    # Seuils pour la détection structurelle
    min_contour_area = diff_conf.get("min_contour_area", 100)
    min_contour_dim = diff_conf.get("min_contour_dim", 10)
    max_added_surface = diff_conf.get("max_added_surface", 2000)  # pixels
    
    # Seuils de fallback (méthode classique)
    max_hard = diff_conf.get("max_hard_pixels", 12000)
    thr_hard = diff_conf.get("threshold_hard", 80)
    
    # 1. Convertir en gris
    a_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    
    # 2. Appliquer les exclusions
    a_gray = apply_exclusions(a_gray.copy(), config)
    r_gray = apply_exclusions(r_gray.copy(), config)
    
    # 3. Binariser si pas déjà fait
    _, a_bin = cv2.threshold(a_gray, 127, 255, cv2.THRESH_BINARY)
    _, r_bin = cv2.threshold(r_gray, 127, 255, cv2.THRESH_BINARY)
    
    # 4. Inverser (texte en blanc)
    a_inv = 255 - a_bin
    r_inv = 255 - r_bin
    
    # 5. Dilater pour tolérer les petits décalages et épaissir le texte
    kernel = np.ones((3, 3), np.uint8)
    a_dilated = cv2.dilate(a_inv, kernel, iterations=2)
    r_dilated = cv2.dilate(r_inv, kernel, iterations=2)
    
    # 6. Ce qui est dans l'aligné mais PAS dans la référence = ajouts
    added = cv2.subtract(a_dilated, r_dilated)
    
    # 7. Érosion pour enlever le bruit fin
    added_clean = cv2.erode(added, kernel, iterations=1)
    
    # 8. Trouver les contours des zones ajoutées
    contours, _ = cv2.findContours(added_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 9. Filtrer les contours significatifs
    significant_contours = []
    total_added_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Ignorer les éléments trop petits (bruit)
        if w < min_contour_dim and h < min_contour_dim:
            continue
        
        # Ignorer les lignes très fines et longues (artefacts de scan)
        if (w > 200 and h < 10) or (h > 200 and w < 10):
            continue
        
        significant_contours.append(cnt)
        total_added_area += area
    
    # 10. Calculer aussi les métriques classiques (pour compatibilité)
    diff_classic = cv2.absdiff(a_gray, r_gray)
    hard_pixels = int(np.sum(diff_classic > thr_hard))
    soft_pixels = int(np.sum(diff_classic > 40))
    
    # 11. Décision
    n_zones = len(significant_contours)
    
    if n_zones > 0 and total_added_area > max_added_surface:
        status = "modifie"
    elif n_zones > 3:  # Plus de 3 zones distinctes = suspect
        status = "modifie"
    else:
        status = "pas-modifie"
    
    # Debug: sauvegarder la carte des ajouts
    if debug_path is not None:
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        debug_img = cv2.cvtColor(a_bin, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_img, significant_contours, -1, (0, 0, 255), 2)
        cv2.imwrite(debug_path, debug_img)
        
        # Sauvegarder aussi la carte d'ajouts brute
        added_path = debug_path.replace("diff_", "added_")
        cv2.imwrite(added_path, added_clean)
    
    logger.debug(f"Structural: zones={n_zones}, surface={total_added_area}px | "
                 f"Classic: soft={soft_pixels}, hard={hard_pixels}")
    
    return status, {
        "zones_added": n_zones,
        "surface_added": int(total_added_area),
        "soft": soft_pixels,
        "hard": hard_pixels
    }


def compare_images(aligned, ref_img, config, debug_path=None, is_snl=False):
    """
    Fonction de comparaison principale.
    Utilise la méthode structurelle par défaut.
    """
    return compare_images_structural(aligned, ref_img, config, debug_path, is_snl)


def process_one(image_path, export_root):
    """
    Traite une seule image de bulletin.
    
    Workflow:
    1. Charger l'image originale (couleur)
    2. Identifier le numéro de liste
    3. Charger la référence correspondante
    4. Redimensionner et convertir l'image en N&B
    5. Détecter les coins et aligner
    6. Comparer avec la méthode structurelle
    7. Copier l'image ORIGINALE vers le bon dossier
    """
    config = load_config()

    # 1. Charger l'image originale (potentiellement couleur)
    src_img_original = cv2.imread(image_path)
    if src_img_original is None:
        logger.error(f"Impossible de lire {image_path}")
        return

    filename = os.path.basename(image_path)
    logger.info(f"Traitement de {filename}")

    # 2. Convertir en N&B pour l'identification du numéro
    src_img_bw = convert_for_comparison(src_img_original, config)
    if src_img_bw is None:
        logger.error(f"Échec conversion N&B: {filename}")
        return

    # 3. Identifier le numéro de liste
    num1, ref1 = identify_list_number(src_img_bw, config, debug=False, no_snl=True)
    logger.debug(f"{filename} essai1: num={num1}, ref={ref1}")

    if num1 is not None and ref1 is not None:
        num_liste, ref_name = num1, ref1
    else:
        num2, ref2 = identify_list_number(src_img_bw, config, debug=False, no_snl=False)
        logger.debug(f"{filename} essai2: num={num2}, ref={ref2}")

        if num2 is not None and ref2 is not None:
            num_liste, ref_name = num2, ref2
        else:
            ref_name = config["lists"].get("SNL", None)
            num_liste = "SNL"
            logger.warning(f"{filename}: fallback SNL")

    if ref_name is None:
        logger.error(f"{filename}: pas de référence disponible")
        return

    # 4. Charger la référence
    ref_dir = config["paths"]["ref_bulletins_dir"]
    ref_path = os.path.join(ref_dir, ref_name)
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        logger.error(f"Référence introuvable: {ref_path}")
        return

    # 5. NOUVEAU: Redimensionner ET convertir pour matcher la référence
    src_img_prepared = convert_and_resize_for_comparison(src_img_original, ref_img, config)
    if src_img_prepared is None:
        logger.error(f"Échec préparation: {filename}")
        return

    # 6. Détecter les coins
    src_corners = detect_corners(src_img_prepared, config, debug_name=filename)
    ref_corners = detect_corners(ref_img, config, debug_name=ref_name)

    # 7. Aligner
    aligned = align_images(src_img_prepared, ref_img, src_corners, ref_corners)

    # 8. Comparer (méthode structurelle)
    debug_diff_path = os.path.join(export_root, "align", f"diff_{filename}")
    is_snl = ref_name.startswith("Bulletin_SNL")
    status, counts = compare_images(
        aligned,
        ref_img,
        config,
        debug_path=debug_diff_path,
        is_snl=is_snl
    )

    logger.info(f"{filename} → {status} "
                f"(zones={counts['zones_added']}, surface={counts['surface_added']}px) "
                f"ref={ref_name}")

    # 9. Créer les dossiers d'export
    align_dir = os.path.join(export_root, "align")
    mod_dir = os.path.join(export_root, "modifie")
    ok_dir = os.path.join(export_root, "pas-modifie")
    os.makedirs(align_dir, exist_ok=True)
    os.makedirs(mod_dir, exist_ok=True)
    os.makedirs(ok_dir, exist_ok=True)

    # 10. Sauvegarder l'image alignée (N&B, pour debug)
    aligned_path = os.path.join(align_dir, filename)
    cv2.imwrite(aligned_path, aligned)

    # 11. IMPORTANT: Copier l'image ORIGINALE (couleur!)
    if status == "modifie":
        copyfile(image_path, os.path.join(mod_dir, filename))
    else:
        copyfile(image_path, os.path.join(ok_dir, filename))

    # 12. Rapport
    append_report(export_root, [
        filename,
        num_liste,
        ref_name,
        counts["zones_added"],
        counts["surface_added"],
        counts["soft"],
        counts["hard"],
        status
    ])


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
    
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    else:
        test_img = os.path.join(BASE_DIR, "..", "bulletins-a-trier", "test.png")
    
    process_one(test_img, export_root="./export_test")