# src/align_and_compare.py (extrait)
import os
import csv
import cv2
import json
import numpy as np
from shutil import copyfile

from src.detect_corners import detect_corners
from src.identify_list_number import identify_list_number

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "..", "config", "config.json")

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
                    "pixels_diff_soft",
                    "pixels_diff_hard",
                    "statut"
                ])
            writer.writerow(row)
    except PermissionError:
        alt_path = os.path.join(export_root, "report_backup.csv")
        print(f"[WARN] report.csv verrouillé, écriture dans {alt_path}")
        with open(alt_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            if not os.path.exists(alt_path) or os.path.getsize(alt_path) == 0:
                writer.writerow([
                    "fichier",
                    "numero_liste",
                    "template",
                    "pixels_diff_soft",
                    "pixels_diff_hard",
                    "statut"
                ])
            writer.writerow(row)

def apply_exclusions(img, config):
    ex = config["zones"].get("exclusion", [])
    for zone in ex:
        x, y, w, h = [int(zone[k]) for k in ("x", "y", "w", "h")]
        img[y:y+h, x:x+w] = 255
    return img

def align_images(src_img, ref_img, src_corners, ref_corners):
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

def compare_images(aligned, ref_img, config, debug_path=None, is_snl=False):
    diff_conf = config.get("diff", {})
    thr_soft = diff_conf.get("threshold_soft", 40)
    max_soft = diff_conf.get("max_soft_pixels", 500000)

    thr_hard = diff_conf.get("threshold_hard", 80)
    max_hard = diff_conf.get("max_hard_pixels", 12000)
    min_blob_area = diff_conf.get("min_hard_blob_area", 200)

    # valeurs par défaut (pour les listes de parti)
    excl_top = diff_conf.get("hard_exclude_top", 220)
    excl_bottom = diff_conf.get("hard_exclude_bottom", 220)

    # MAIS pour le SNL on ne veut PAS exclure le haut/bas
    if is_snl:
        excl_top = 0
        excl_bottom = 0

    # 1) gris
    a_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    # exclusions communes (codes imprimés etc.)
    a_gray = apply_exclusions(a_gray, config)
    r_gray = apply_exclusions(r_gray, config)

    h, w = a_gray.shape

    # ---------- PASSE SOFT ----------
    a_soft = cv2.equalizeHist(a_gray)
    r_soft = cv2.equalizeHist(r_gray)
    a_soft = cv2.GaussianBlur(a_soft, (5, 5), 0)
    r_soft = cv2.GaussianBlur(r_soft, (5, 5), 0)

    diff_soft = cv2.absdiff(a_soft, r_soft)
    mask_soft = (diff_soft > thr_soft).astype(np.uint8)
    count_soft = int(mask_soft.sum())

    # ---------- PASSE HARD ----------
    a_h = cv2.GaussianBlur(a_gray, (5, 5), 0)
    r_h = cv2.GaussianBlur(r_gray, (5, 5), 0)
    diff_hard = cv2.absdiff(a_h, r_h)

    # on coupe le haut/bas seulement si ce n’est pas un SNL
    if excl_top > 0:
        diff_hard[:excl_top, :] = 0
    if excl_bottom > 0:
        diff_hard[h - excl_bottom:h, :] = 0

    _, hard_bin = cv2.threshold(diff_hard, thr_hard, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(hard_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hard_pixels = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_blob_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # ignorer les petits trucs plats (texte imprimé décalé)
        if bh < 20:
            continue

        # ignorer les très longues lignes fines
        if bw > 200 and bh < 40:
            continue

        hard_pixels += int(area)

    if debug_path is not None:
        dbg = (mask_soft * 255).astype("uint8")
        cv2.imwrite(debug_path, dbg)

    # décision
    if hard_pixels > max_hard:
        status = "modifie"
    elif count_soft > max_soft:
        status = "modifie"
    else:
        status = "pas-modifie"

    print(f"[DBG] soft={count_soft}px (>{max_soft}?) | hard_center={hard_pixels}px (>{max_hard}?) | is_snl={is_snl}")

    return status, {"soft": count_soft, "hard": hard_pixels}

def process_one(image_path, export_root):
    config = load_config()

    src_img = cv2.imread(image_path)
    if src_img is None:
        print("[ERR] impossible de lire", image_path)
        return

    filename = os.path.basename(image_path)

    # --- 1) ESSAI 1 : on force la lecture d’un numéro, même si la zone ressemble à SNL
    num1, ref1 = identify_list_number(src_img, config, debug=False, no_snl=True)
    print(f"[DBG] {filename} essai1(no_snl=True) → num={num1}, ref={ref1}")

    # si on a un numéro ET un modèle, on prend ça tout de suite
    if num1 is not None and ref1 is not None:
        num_liste, ref_name = num1, ref1
    else:
        # --- 2) ESSAI 2 : on laisse la détection SNL normale
        num2, ref2 = identify_list_number(src_img, config, debug=False, no_snl=False)
        print(f"[DBG] {filename} essai2(no_snl=False) → num={num2}, ref={ref2}")

        if num2 is not None and ref2 is not None:
            num_liste, ref_name = num2, ref2
        else:
            # --- 3) FALLBACK : SNL costaud
            ref_name = config["lists"].get("SNL", None)
            num_liste = "SNL"
            print(f"[WARN] {filename} : aucun numero lisible → fallback SNL")

    if ref_name is None:
        print(f"[ERR] {filename} : pas de bulletin de référence disponible (même pas SNL), on saute.")
        return

    # 2) charger l'image de référence
    ref_dir = config["paths"]["ref_bulletins_dir"]
    ref_path = os.path.join(ref_dir, ref_name)
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print("[ERR] impossible de lire le bulletin de reference", ref_path)
        return

    # 3) coins
    src_corners = detect_corners(src_img, config, debug_name=filename)
    ref_corners = detect_corners(ref_img, config, debug_name=ref_name)

    # 4) aligner
    aligned = align_images(src_img, ref_img, src_corners, ref_corners)

    # 5) comparer
    # status, changed_count, diff = compare_images(aligned, ref_img, config)
    debug_diff_path = os.path.join(export_root, "align", f"diff_{filename}")
    is_snl = ref_name.startswith("Bulletin_SNL")
    status, counts = compare_images(
        aligned,
        ref_img,
        config,
        debug_path=debug_diff_path,
        is_snl=is_snl
    )
    print(f"[INFO] {filename} → {status} (soft={counts['soft']}, hard={counts['hard']}) ref={ref_name}")

    # 6) dossiers d’export
    align_dir = os.path.join(export_root, "align")
    mod_dir = os.path.join(export_root, "modifie")
    ok_dir = os.path.join(export_root, "pas-modifie")
    os.makedirs(align_dir, exist_ok=True)
    os.makedirs(mod_dir, exist_ok=True)
    os.makedirs(ok_dir, exist_ok=True)

    # 7) sauver l’aligné
    cv2.imwrite(os.path.join(align_dir, filename), aligned)

    # 8) copier l’original
    from shutil import copyfile
    if status == "modifie":
        copyfile(image_path, os.path.join(mod_dir, filename))
    else:
        copyfile(image_path, os.path.join(ok_dir, filename))

    # 9) rapport
    append_report(export_root, [
        filename,
        num_liste,
        ref_name,
        counts["soft"],   # pixels différents après passe douce
        counts["hard"],   # pixels différents après passe brute
        status
    ])

    print(f"[INFO] {filename} → {status} (soft={counts['soft']}, hard={counts['hard']}) ref={ref_name}")

if __name__ == "__main__":
    # test sur une seule image
    test_img = os.path.join(BASE_DIR, "..", "bulletins-a-trier", "Bulletin_PLR_003.png")
    process_one(test_img)
