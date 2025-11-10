#!/usr/bin/env python3
"""
Outil de visualisation combinée des zones configurées.
Affiche les zones d'exclusion ET la zone d'identification sur une image de base.
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path


def visualize_all_zones(image_path: str, config_path: str = "ref/config.json"):
    """
    Visualise toutes les zones configurées sur une image de base.
    
    Args:
        image_path: Chemin de l'image de base
        config_path: Chemin du fichier de configuration
    """
    print("\n" + "="*70)
    print("VISUALISATION DES ZONES CONFIGURÉES")
    print("="*70 + "\n")
    
    # Charger la configuration
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✓ Configuration chargée depuis {config_path}")
    except FileNotFoundError:
        print(f"✗ Fichier de configuration non trouvé: {config_path}")
        return
    
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"✗ Impossible de charger l'image: {image_path}")
        return
    
    print(f"✓ Image chargée: {image_path}")
    print(f"  Dimensions: {image.shape[1]} × {image.shape[0]} pixels\n")
    
    # Créer une copie pour la visualisation
    vis_image = image.copy()
    
    # Statistiques
    exclusion_count = 0
    id_zone_present = False
    zones_out_of_bounds = []
    
    # 1. Dessiner les zones d'exclusion (en rouge)
    exclusion_zones = config.get('exclusion_zones', [])
    print(f"→ Zones d'exclusion: {len(exclusion_zones)}")
    
    for zone in exclusion_zones:
        x, y, w, h = zone['x'], zone['y'], zone['width'], zone['height']
        name = zone.get('name', 'Sans nom')
        
        # Vérifier si dans les limites
        img_h, img_w = image.shape[:2]
        if x >= 0 and y >= 0 and x + w <= img_w and y + h <= img_h:
            # Dessiner le rectangle (rouge)
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
            
            # Ajouter le nom (fond blanc pour lisibilité)
            text = f"EXCLUSION: {name}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_image, (x, y - text_h - 10), (x + text_w + 10, y), (255, 255, 255), -1)
            cv2.putText(vis_image, text, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Ajouter les dimensions
            dim_text = f"{w}×{h}"
            cv2.putText(vis_image, dim_text, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            exclusion_count += 1
            print(f"  ✓ {name}: ({x}, {y}) {w}×{h} pixels")
        else:
            zones_out_of_bounds.append(name)
            print(f"  ✗ {name}: HORS LIMITES!")
    
    # 2. Dessiner la zone d'identification (en vert)
    id_zone = config.get('identification_zone', {})
    if id_zone.get('enabled', False):
        print(f"\n→ Zone d'identification:")
        
        x, y, w, h = id_zone['x'], id_zone['y'], id_zone['width'], id_zone['height']
        threshold = id_zone.get('black_pixel_threshold_percent', 5.0)
        
        # Vérifier si dans les limites
        img_h, img_w = image.shape[:2]
        if x >= 0 and y >= 0 and x + w <= img_w and y + h <= img_h:
            # Dessiner le rectangle (vert)
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Ajouter le label (fond blanc)
            text = "IDENTIFICATION"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(vis_image, (x, y - text_h - 10), (x + text_w + 10, y), (255, 255, 255), -1)
            cv2.putText(vis_image, text, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Ajouter les dimensions et seuil
            dim_text = f"{w}×{h} | Seuil: {threshold}%"
            cv2.putText(vis_image, dim_text, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Analyser le contenu de la zone
            zone_img = image[y:y+h, x:x+w]
            gray = cv2.cvtColor(zone_img, cv2.COLOR_BGR2GRAY) if len(zone_img.shape) == 3 else zone_img
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            inverted = cv2.bitwise_not(binary)
            
            black_pixels = np.count_nonzero(inverted)
            total_pixels = zone_img.size if len(zone_img.shape) == 2 else zone_img.shape[0] * zone_img.shape[1]
            black_percent = (black_pixels / total_pixels) * 100
            
            # Déterminer le type
            if black_percent >= threshold:
                detected_type = "AVEC_NUMERO"
                type_color = (0, 255, 0)
            else:
                detected_type = "SNL"
                type_color = (0, 165, 255)
            
            id_zone_present = True
            print(f"  ✓ Position: ({x}, {y}) {w}×{h} pixels")
            print(f"  ✓ Pixels noirs: {black_percent:.2f}% (seuil: {threshold}%)")
            print(f"  ✓ Type détecté: {detected_type}")
            
            # Ajouter le type détecté sur l'image
            type_text = f"Type: {detected_type}"
            cv2.putText(vis_image, type_text, (x, y - text_h - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, type_color, 2)
        else:
            zones_out_of_bounds.append("Zone d'identification")
            print(f"  ✗ HORS LIMITES!")
            print(f"    Zone demandée: ({x}, {y}) → ({x+w}, {y+h})")
            print(f"    Image: (0, 0) → ({img_w}, {img_h})")
    else:
        print(f"\n→ Zone d'identification: DÉSACTIVÉE")
    
    # 3. Afficher les statistiques
    print(f"\n{'='*70}")
    print("RÉSUMÉ")
    print(f"{'='*70}")
    print(f"✓ {exclusion_count} zone(s) d'exclusion valide(s)")
    print(f"✓ Zone d'identification: {'CONFIGURÉE' if id_zone_present else 'NON CONFIGURÉE'}")
    
    if zones_out_of_bounds:
        print(f"\n⚠ {len(zones_out_of_bounds)} zone(s) HORS LIMITES:")
        for zone_name in zones_out_of_bounds:
            print(f"  • {zone_name}")
        print("\n→ Veuillez reconfigurer ces zones avec les bonnes dimensions")
    
    # 4. Créer une légende
    legend = np.zeros((120, vis_image.shape[1], 3), dtype=np.uint8)
    legend[:] = (240, 240, 240)  # Fond gris clair
    
    # Titre
    cv2.putText(legend, "LEGENDE", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Zones d'exclusion (rouge)
    cv2.rectangle(legend, (10, 40), (30, 60), (0, 0, 255), -1)
    cv2.putText(legend, f"Zones d'exclusion ({exclusion_count})", (40, 55), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Zone d'identification (vert)
    cv2.rectangle(legend, (10, 70), (30, 90), (0, 255, 0), -1)
    status = "CONFIGUREE" if id_zone_present else "NON CONFIGUREE"
    cv2.putText(legend, f"Zone d'identification ({status})", (40, 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Avertissement si zones hors limites
    if zones_out_of_bounds:
        cv2.putText(legend, f"⚠ {len(zones_out_of_bounds)} zone(s) HORS LIMITES!", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 5. Assembler image + légende
    final_display = np.vstack([legend, vis_image])
    
    # 6. Afficher
    window_name = "Zones Configurées - Appuyez sur une touche pour fermer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, final_display)
    
    print(f"\n{'='*70}")
    print("VISUALISATION")
    print(f"{'='*70}")
    print("→ Fenêtre affichée avec:")
    print("  • Zones d'exclusion en ROUGE")
    print("  • Zone d'identification en VERT")
    print("  • Dimensions et noms affichés")
    print("\nAppuyez sur une touche dans la fenêtre pour fermer...")
    print(f"{'='*70}\n")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 7. Recommandations finales
    if zones_out_of_bounds:
        print("\n" + "="*70)
        print("⚠ ACTION REQUISE")
        print("="*70)
        print("\nCertaines zones sont hors limites de l'image.")
        print("Pour les reconfigurer:")
        print()
        print("  # Zones d'exclusion:")
        print(f"  python define_exclusions.py {image_path}")
        print()
        print("  # Zone d'identification:")
        print(f"  python define_identification_zone.py {image_path}")
        print()
    elif not id_zone_present and exclusion_count == 0:
        print("\n" + "="*70)
        print("ℹ CONFIGURATION INCOMPLÈTE")
        print("="*70)
        print("\nAucune zone n'est configurée.")
        print("Pour configurer:")
        print()
        print("  # Zones d'exclusion (numéros de document, dates, etc.):")
        print(f"  python define_exclusions.py {image_path}")
        print()
        print("  # Zone d'identification (pour détecter le type de bulletin):")
        print(f"  python define_identification_zone.py {image_path}")
        print()
    else:
        print("\n✅ Configuration complète et valide!\n")


def main():
    """Point d'entrée principal."""
    if len(sys.argv) < 2:
        print("\nUsage: python visualize_zones.py <image_de_base>")
        print("\nExemple:")
        print("  python visualize_zones.py images_base/vernier_SNL.png")
        print("\nCe script permet de:")
        print("  • Visualiser TOUTES les zones configurées")
        print("  • Voir les zones d'exclusion (rouge)")
        print("  • Voir la zone d'identification (vert)")
        print("  • Vérifier que les zones sont dans les limites")
        print("  • Tester la détection du type de bulletin")
        print()
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"✗ Fichier non trouvé: {image_path}")
        sys.exit(1)
    
    visualize_all_zones(image_path)


if __name__ == "__main__":
    main()