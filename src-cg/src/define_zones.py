import os
import json
import cv2
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "..", "config", "config.json")
DEFAULT_IMG = os.path.join(BASE_DIR, "..", "ref", "bulletins", "Bulletin_SNL_000.png")

drawing = False
ix, iy = -1, -1
current_rect = None
zoom = 0.4  # facteur de zoom initial (0.4 = 40% de la taille réelle)

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {
            "zones": {
                "identification": {"x": 1475, "y": 620, "w": 130, "h": 100},
                "zone_scrutin": {"x": 200, "y": 200, "w": 250, "h": 80},
                "exclusion": []
            },
            "corner_boxes": {},
            "corners": {"match_threshold": 0.4},
            "diff": {"threshold": 40, "max_added_pixels": 1200}
        }
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("[INFO] Config sauvegardée dans", CONFIG_PATH)

def mouse_cb(event, x, y, flags, param):
    global ix, iy, drawing, current_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = int(x / zoom), int(y / zoom)
        current_rect = None
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_rect = (ix, iy, int(x / zoom), int(y / zoom))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_rect = (ix, iy, int(x / zoom), int(y / zoom))

def rect_to_xywh(r):
    x1, y1, x2, y2 = r
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return x, y, w, h

def resize_for_display(img):
    global zoom
    h, w = img.shape[:2]
    disp_w = int(w * zoom)
    disp_h = int(h * zoom)
    return cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

def main(image_path):
    global zoom
    cfg = load_config()
    img = cv2.imread(image_path)
    if img is None:
        print("[ERR] Impossible de charger", image_path)
        return

    clone = img.copy()
    cv2.namedWindow("zones", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("zones", mouse_cb)

    exclusions = cfg["zones"].get("exclusion", [])
    identification = cfg["zones"].get("identification", None)

    print("COMMANDES :")
    print("  - E : ajouter zone d'exclusion")
    print("  - I : définir zone d'identification")
    print("  - S : sauvegarder le config")
    print("  - +/- : zoom/dézoom")
    print("  - Q : quitter")

    while True:
        disp = resize_for_display(clone)

        # anciennes zones
        for ex in exclusions:
            x, y, w, h = [int(ex[k]) for k in ("x", "y", "w", "h")]
            cv2.rectangle(disp, (int(x*zoom), int(y*zoom)),
                          (int((x+w)*zoom), int((y+h)*zoom)), (0, 0, 255), 2)
        if identification:
            x, y, w, h = [int(identification[k]) for k in ("x", "y", "w", "h")]
            cv2.rectangle(disp, (int(x*zoom), int(y*zoom)),
                          (int((x+w)*zoom), int((y+h)*zoom)), (0, 255, 0), 2)

        # rectangle courant
        if current_rect is not None:
            x1, y1, x2, y2 = current_rect
            cv2.rectangle(disp, (int(x1*zoom), int(y1*zoom)), (int(x2*zoom), int(y2*zoom)), (255, 0, 0), 1)

        cv2.putText(disp, f"Zoom: {zoom:.2f}x (E/I/S/Q, +/-)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

        cv2.imshow("zones", disp)
        k = cv2.waitKey(20) & 0xFF

        if k in [ord('q'), 27]:
            break
        elif k == ord('e') and current_rect:
            x, y, w, h = rect_to_xywh(current_rect)
            exclusions.append({"x": x, "y": y, "w": w, "h": h, "comment": "exclusion"})
            cfg["zones"]["exclusion"] = exclusions
            print("[INFO] zone d'exclusion ajoutée:", x, y, w, h)
        elif k == ord('i') and current_rect:
            x, y, w, h = rect_to_xywh(current_rect)
            identification = {"x": x, "y": y, "w": w, "h": h}
            cfg["zones"]["identification"] = identification
            print("[INFO] zone d'identification définie:", x, y, w, h)
        elif k == ord('s'):
            save_config(cfg)
        elif k == ord('+'):
            zoom = min(1.5, zoom + 0.1)
        elif k == ord('-'):
            zoom = max(0.2, zoom - 0.1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Définir les zones sur un bulletin de référence")
    parser.add_argument("--image", type=str, default=DEFAULT_IMG,
                        help="chemin de l'image à utiliser")
    args = parser.parse_args()
    main(args.image)
