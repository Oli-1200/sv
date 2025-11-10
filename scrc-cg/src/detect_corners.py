import os
import json
import cv2
import logging
import numpy as np

# chemins relatifs basés sur ce fichier
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../src
CONFIG_PATH = os.path.join(BASE_DIR, "..", "config", "config.json")
CORNERS_DIR = os.path.join(BASE_DIR, "..", "ref", "corners")

logger = logging.getLogger(__name__)

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_corner_template(name):
    path = os.path.join(CORNERS_DIR, f"{name}.png")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Template de coin manquant : {path}")
    tpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return tpl

def preprocess_bin(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def preprocess_edge(img):
    return cv2.Canny(img, 50, 150)

def match_with_fallback(roi, tpl, min_ok=0.4):
    # 1) binaire
    roi_bin = preprocess_bin(roi)
    tpl_bin = preprocess_bin(tpl)
    res = cv2.matchTemplate(roi_bin, tpl_bin, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val >= min_ok:
        return max_val, max_loc

    # 2) edges
    roi_e = preprocess_edge(roi)
    tpl_e = preprocess_edge(tpl)
    res2 = cv2.matchTemplate(roi_e, tpl_e, cv2.TM_CCOEFF_NORMED)
    _, max_val2, _, max_loc2 = cv2.minMaxLoc(res2)

    if max_val2 > max_val:
        return max_val2, max_loc2
    else:
        return max_val, max_loc

def detect_corners(image_bgr, config, debug_name=None):
    """
    Détecte les coins en respectant les limites de l'image.
    debug_name sert juste à sortir le nom de l'image dans les logs.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape[:2]

    if debug_name:
        logger.debug(f"detect_corners sur {debug_name} ({w_img}x{h_img})")

    boxes = config["corner_boxes"]
    threshold = config["corners"].get("match_threshold", 0.4)
    corners_found = {}

    for name, box in boxes.items():
        # valeurs du config
        x = int(box["x"])
        y = int(box["y"])
        w = int(box["w"])
        h = int(box["h"])

        # ✅ recadrage dans l'image
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = min(w, w_img - x)
        h = min(h, h_img - y)

        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            logger.warning(f"Zone vide pour {name} dans {debug_name or ''} : x={x}, y={y}, w={w}, h={h}")
            # on met un coin par défaut pour ne pas bloquer
            corners_found[name] = (x, y)
            continue

        tpl = load_corner_template(name)

        max_val, max_loc = match_with_fallback(roi, tpl, min_ok=threshold)

        if max_val < threshold:
            logger.warning(f"Corrélation faible pour {name} ({debug_name or ''}): {max_val:.2f}")

        # position dans l'image entière
        corners_found[name] = (x + max_loc[0], y + max_loc[1])

    return corners_found

def detect_orientation(image_bgr, config, ref_patch=None):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    z = config["zones"]["zone_scrutin"]
    x, y, w, h = [int(z[k]) for k in ("x", "y", "w", "h")]
    patch = gray[y:y+h, x:x+w]

    if ref_patch is None:
        return "upright", patch

    patch_rot = cv2.rotate(patch, cv2.ROTATE_180)
    ref_rot = cv2.rotate(ref_patch, cv2.ROTATE_180)

    corr_u = cv2.matchTemplate(patch, ref_patch, cv2.TM_CCOEFF_NORMED)
    corr_r = cv2.matchTemplate(patch, ref_rot, cv2.TM_CCOEFF_NORMED)
    val_u = float(np.max(corr_u))
    val_r = float(np.max(corr_r))

    orient = "upright" if val_u >= val_r else "rotated"
    return orient, ref_patch

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
    config = load_config()
    test_img_path = os.path.join(BASE_DIR, "..", "ref", "bulletins", "Bulletin_SNL_000.png")
    img = cv2.imread(test_img_path)
    corners = detect_corners(img, config, debug_name="Bulletin_SNL_000.png")
    logger.info(f"Coins détectés : {corners}")
    orient, _ = detect_orientation(img, config)
    logger.info(f"Orientation : {orient}")