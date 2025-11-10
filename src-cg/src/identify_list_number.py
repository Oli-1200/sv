# src/identify_list_number.py
import os
import json
import cv2
import numpy as np

# chemins de base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../src
CONFIG_PATH = os.path.join(BASE_DIR, "..", "config", "config.json")
DIGITS_DIR = os.path.join(BASE_DIR, "..", "ref", "digits")


# ------------------------------------------------------------
# Chargement config
# ------------------------------------------------------------
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# Chargement des templates de chiffres
# ------------------------------------------------------------
def load_digit_templates(target_size=32):
    """
    Charge 0-9 depuis ref/digits, binarise, recadre et redimensionne
    en target_size x target_size, stocké en 0/1.
    """
    templates = {}
    for d in range(10):
        path = os.path.join(DIGITS_DIR, f"{d}.png")
        if not os.path.exists(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # binarisation
        _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # recadre autour du contenu
        img_bin = crop_to_content(img_bin)
        # redimensionne
        tpl_resized = cv2.resize(img_bin, (target_size, target_size), interpolation=cv2.INTER_AREA)
        # 0/1
        tpl01 = (tpl_resized < 128).astype(np.uint8)
        templates[str(d)] = tpl01
    if not templates:
        raise FileNotFoundError("Aucun template de chiffre trouvé dans ref/digits/")
    return templates


# ------------------------------------------------------------
# Utilitaires d'image
# ------------------------------------------------------------
def crop_identification_zone(img_bgr, config):
    z = config["zones"]["identification"]
    x, y, w, h = int(z["x"]), int(z["y"]), int(z["w"]), int(z["h"])
    return img_bgr[y:y+h, x:x+w].copy()


def binarize(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # améliore le contraste un peu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def crop_to_content(binary_img, pad=2):
    """
    Coupe les grosses marges blanches autour d’un chiffre (image binaire 0/255).
    """
    ys, xs = np.where(binary_img == 0)
    if len(xs) == 0 or len(ys) == 0:
        return binary_img
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_min = max(x_min - pad, 0)
    y_min = max(y_min - pad, 0)
    x_max = min(x_max + pad, binary_img.shape[1] - 1)
    y_max = min(y_max + pad, binary_img.shape[0] - 1)
    return binary_img[y_min:y_max+1, x_min:x_max+1]


# ------------------------------------------------------------
# Détection SNL
# ------------------------------------------------------------
def detect_snl(binary_zone):
    """
    Détection SNL plus permissive :
    - on regarde surtout la bande du bas : si une ligne noire bien marquée,
      on considère que c'est SNL, même s'il y a du noir au-dessus (écriture).
    """
    h, w = binary_zone.shape
    bottom_band = binary_zone[int(h * 0.7):h, :]
    black_bottom = np.sum(bottom_band < 10)

    # seuil à ajuster selon épaisseur du trait
    if black_bottom > w * 3:
        return True
    return False


# ------------------------------------------------------------
# Segmentation des chiffres
# ------------------------------------------------------------
def segment_digits_by_contours(binary_zone, debug=False):
    """
    Retourne une liste d'images binaires (0/255) des chiffres, triées de gauche à droite.
    """
    # petit nettoyage
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(binary_zone, cv2.MORPH_OPEN, kernel, iterations=1)

    # on inverse pour trouver des objets blancs
    inv = 255 - clean
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    digits = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # on vire le bruit
        if w < 10 or h < 20:
            continue
        roi = binary_zone[y:y+h, x:x+w]
        digits.append((x, roi))
        boxes.append((x, y, w, h))

    digits.sort(key=lambda t: t[0])
    digit_imgs = [roi for _, roi in digits]

    if debug:
        dbg = cv2.cvtColor(binary_zone, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in boxes:
            cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.imshow("digits_by_contours", dbg)

    return digit_imgs


# ------------------------------------------------------------
# Matching binaire pixel-à-pixel
# ------------------------------------------------------------
def match_digit_pixelwise(digit_img, templates, target_size=32):
    """
    digit_img : image binaire 0/255 d’un seul chiffre
    templates : dict "digit" -> image 0/1 target_size x target_size
    Retourne (digit, score_similarité) avec score entre 0 et 1
    """
    digit_img = crop_to_content(digit_img)
    resized = cv2.resize(digit_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    digit01 = (resized < 128).astype(np.uint8)

    best_digit = None
    best_score = -1.0

    for d, tpl01 in templates.items():
        diff = np.bitwise_xor(digit01, tpl01)
        diff_pixels = diff.sum()
        total_pixels = diff.size
        score = 1.0 - (diff_pixels / float(total_pixels))
        if score > best_score:
            best_score = score
            best_digit = d

    return best_digit, best_score


# ------------------------------------------------------------
# Fonction principale de détection
# ------------------------------------------------------------
def identify_list_number(img_bgr, config, debug=False, no_snl=False):
    """
    Retourne (code_liste, nom_fichier_ref) ou (None, None)
    """
    zone = crop_identification_zone(img_bgr, config)
    bin_zone = binarize(zone)

    if debug:
        cv2.imshow("zone_identification", zone)
        cv2.imshow("zone_identification_bin", bin_zone)

    # 1) cas SNL (sauf si on a demandé de l'ignorer)
    if not no_snl and detect_snl(bin_zone):
        ref_name = config["lists"].get("SNL", "Bulletin_SNL_000.png")
        if debug:
            print("[DEBUG] SNL détecté")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return "SNL", ref_name

    # 2) sinon on essaie de lire les chiffres
    digit_imgs = segment_digits_by_contours(bin_zone, debug=debug)

    if debug:
        print(f"[DEBUG] {len(digit_imgs)} chiffre(s) trouvé(s)")
        for i, dimg in enumerate(digit_imgs):
            cv2.imshow(f"digit_{i+1}", dimg)

    if not digit_imgs:
        if debug:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return None, None

    templates = load_digit_templates(target_size=32)

    found = []
    for i, dimg in enumerate(digit_imgs):
        digit, score = match_digit_pixelwise(dimg, templates, target_size=32)
        if debug:
            print(f"[DEBUG] digit {i+1} → {digit} (score={score:.2f})")
        # seuil à 0.55, à ajuster selon la qualité
        if score > 0.55:
            found.append(digit)
        else:
            found.append(None)

    if debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # reconstruire
    clean_digits = [d for d in found if d is not None]
    if not clean_digits:
        return None, None

    code = "".join(clean_digits)

    # normaliser sur 2 chiffres
    if len(code) == 1:
        code = code.zfill(2)

    ref_name = config["lists"].get(code, None)
    return code, ref_name


# ------------------------------------------------------------
# Exécution en ligne de commande
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test de détection du numéro de liste")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Chemin de l'image à tester (sinon prend une image de référence)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Affiche les étapes intermédiaires"
    )
    parser.add_argument(
        "--no-snl",
        action="store_true",
        help="Désactiver la détection automatique du bulletin SNL"
    )
    args = parser.parse_args()

    # chemins
    default_img = os.path.join(BASE_DIR, "..", "ref", "bulletins", "Bulletin_PLR_003.png")
    img_path = args.image or default_img

    if not os.path.exists(img_path):
        print(f"[ERR] image introuvable : {img_path}")
        exit(1)

    config = load_config()
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERR] impossible de lire {img_path}")
        exit(1)

    code, ref = identify_list_number(img, config, debug=args.debug, no_snl=args.no_snl)
    print("Liste détectée :", code)
    print("Fichier de référence :", ref)
