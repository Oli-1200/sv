#!/usr/bin/env python3
"""
Script de test pour visualiser et ajuster la zone d'identification du type de bulletin.
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path


def visualize_identification_zone(image_path: str, config_path: str = "ref/config.json"):
    """
    Visualise la zone d'identification sur une image et analyse son contenu.
    
    Args:
        image_path: Chemin de l'image à tester
        config_path: Chemin du fichier de configuration
    """
    print("\n" + "="*70)
    print("TEST DE LA ZONE D'IDENTIFICATION")
    print("="*70 + "\n")
    
    # Charger la configuration
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✓ Configuration chargée depuis {config_path}")
    except FileNotFoundError:
        print(f"✗ Fichier de configuration non trouvé: {config_path}")
        return
    
    # Vérifier que la zone d'identification est configurée
    if 'identification_zone' not in config:
        print("✗ Aucune zone d'identification définie dans la configuration")
        return
    
    id_zone = config['identification_zone']
    
    if not id_zone.get('enabled', False):
        print("⚠ La zone d'identification est désactivée dans la configuration")
        print("  Pour l'activer, mettez 'enabled': true dans config.json")
        return
    
    print(f"✓ Zone d'identification configurée:")
    print(f"  • Position: ({id_zone['x']}, {id_zone['y']})")
    print(f"  • Dimensions: {id_zone['width']} x {id_zone['height']} pixels")
    print(f"  • Seuil: {id_zone.get('black_pixel_threshold_percent', 5.0)}%")
    print()
    
    # Charger l'image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"✗ Impossible de charger l'image: {image_path}")
        return
    
    print(f"✓ Image chargée: {image_path}")
    print(f"  • Dimensions: {image.shape[1]} x {image.shape[0]} pixels")
    print()
    
    # Extraire les paramètres de la zone
    x, y, w, h = id_zone['x'], id_zone['y'], id_zone['width'], id_zone['height']
    threshold_percent = id_zone.get('black_pixel_threshold_percent', 5.0)
    binarization_threshold = config.get('processing', {}).get('binarization_threshold', 127)
    
    # Vérifier que la zone est dans les limites
    img_h, img_w = image.shape
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        print(f"✗ ERREUR: Zone d'identification hors limites de l'image!")
        print(f"  Zone demandée: ({x}, {y}) à ({x+w}, {y+h})")
        print(f"  Image: (0, 0) à ({img_w}, {img_h})")
        return
    
    # Extraire la zone
    zone = image[y:y+h, x:x+w]
    
    # Binariser
    _, binary = cv2.threshold(zone, binarization_threshold, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(binary)
    
    # Calculer le pourcentage de pixels noirs
    total_pixels = zone.size
    black_pixels = np.count_nonzero(inverted)
    black_percent = (black_pixels / total_pixels) * 100
    
    # Déterminer le type
    if black_percent >= threshold_percent:
        bulletin_type = "AVEC_NUMERO"
        type_color = (0, 255, 0)  # Vert
    else:
        bulletin_type = "SNL"
        type_color = (0, 165, 255)  # Orange
    
    print("→ Analyse de la zone:")
    print(f"  • Pixels totaux: {total_pixels}")
    print(f"  • Pixels noirs: {black_pixels}")
    print(f"  • Pourcentage noir: {black_percent:.2f}%")
    print(f"  • Seuil configuré: {threshold_percent}%")
    print()
    print(f"✓ Type détecté: {bulletin_type}")
    print()
    
    # Créer les visualisations
    
    # 1. Image complète avec rectangle de la zone
    vis_full = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis_full, (x, y), (x + w, y + h), type_color, 3)
    cv2.putText(vis_full, f"Zone ID: {bulletin_type}", 
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, type_color, 2)
    
    # 2. Zone extraite (agrandie)
    zone_enlarged = cv2.resize(zone, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    zone_color = cv2.cvtColor(zone_enlarged, cv2.COLOR_GRAY2BGR)
    
    # 3. Zone binarisée (agrandie)
    binary_enlarged = cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    binary_color = cv2.cvtColor(binary_enlarged, cv2.COLOR_GRAY2BGR)
    
    # 4. Zone inversée (agrandie)
    inverted_enlarged = cv2.resize(inverted, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    inverted_color = cv2.cvtColor(inverted_enlarged, cv2.COLOR_GRAY2BGR)
    
    # Ajouter du texte sur les zones agrandies
    cv2.putText(zone_color, "Zone originale", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(binary_color, "Binarisee", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(inverted_color, f"Noirs: {black_percent:.1f}%", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Créer une grille de visualisation
    # Redimensionner l'image complète pour l'affichage
    scale = 0.5
    vis_full_resized = cv2.resize(vis_full, None, fx=scale, fy=scale)
    
    # Empiler les zones agrandies
    zones_stack = np.vstack([zone_color, binary_color, inverted_color])
    
    # Ajuster la hauteur pour correspondre
    target_height = vis_full_resized.shape[0]
    current_height = zones_stack.shape[0]
    
    if current_height < target_height:
        # Ajouter un padding noir en bas
        padding = np.zeros((target_height - current_height, zones_stack.shape[1], 3), dtype=np.uint8)
        zones_stack = np.vstack([zones_stack, padding])
    elif current_height > target_height:
        # Couper
        zones_stack = zones_stack[:target_height, :, :]
    
    # Combiner horizontalement
    display = np.hstack([vis_full_resized, zones_stack])
    
    # Ajouter une barre d'information en haut
    info_bar = np.zeros((60, display.shape[1], 3), dtype=np.uint8)
    
    # Texte de résultat
    result_text = f"Type: {bulletin_type} | Noir: {black_percent:.2f}% | Seuil: {threshold_percent}%"
    cv2.putText(info_bar, result_text, 
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, type_color, 2)
    
    # Assembler tout
    final_display = np.vstack([info_bar, display])
    
    # Afficher
    window_name = "Zone d'Identification - Appuyez sur une touche pour fermer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, final_display)
    
    print("→ Fenêtre de visualisation affichée")
    print("  • Gauche: Image complète avec zone encadrée")
    print("  • Droite: Zoom 3x sur la zone (original, binarisé, pixels noirs)")
    print()
    print("Appuyez sur une touche dans la fenêtre pour fermer...")
    print("="*70 + "\n")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Recommandations
    print("\n" + "="*70)
    print("RECOMMANDATIONS")
    print("="*70 + "\n")
    
    if bulletin_type == "AVEC_NUMERO" and black_percent < threshold_percent * 1.5:
        print("⚠ Le pourcentage de noir est proche du seuil.")
        print("  Si ce bulletin est mal classé, envisagez:")
        print(f"  • Diminuer le seuil à {threshold_percent - 1.0}%")
        print("  • Vérifier que la zone englobe bien le numéro")
    
    if bulletin_type == "SNL" and black_percent > threshold_percent * 0.7:
        print("⚠ Le pourcentage de noir est proche du seuil.")
        print("  Si ce bulletin est mal classé, envisagez:")
        print(f"  • Augmenter le seuil à {threshold_percent + 1.0}%")
        print("  • Vérifier que la zone ne contient pas d'autres éléments")
    
    if black_percent < 1.0:
        print("ℹ La zone semble très claire (presque vierge)")
        print("  → Bulletin correctement détecté comme SNL")
    
    if black_percent > 20.0:
        print("ℹ La zone contient beaucoup de noir")
        print("  → Bulletin correctement détecté comme AVEC_NUMERO")
    
    print()


def main():
    """Point d'entrée principal."""
    if len(sys.argv) < 2:
        print("\nUsage: python3 test_identification_zone.py <chemin_image>")
        print("\nExemple:")
        print("  python3 test_identification_zone.py images_source/bulletin_led.png")
        print("  python3 test_identification_zone.py images_source/bulletin_snl.png")
        print("\nCe script permet de:")
        print("  • Visualiser la zone d'identification configurée")
        print("  • Voir le contenu de la zone (agrandi 3x)")
        print("  • Vérifier le type détecté (AVEC_NUMERO ou SNL)")
        print("  • Ajuster les paramètres si nécessaire")
        print("\nLa zone d'identification est configurée dans config.json")
        print()
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"✗ Fichier non trouvé: {image_path}")
        sys.exit(1)
    
    visualize_identification_zone(image_path)


if __name__ == "__main__":
    main()