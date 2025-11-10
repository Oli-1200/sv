#!/usr/bin/env python3
"""
Outil interactif pour définir la zone d'identification sur une image de BASE.
IMPORTANT: Utilisez une IMAGE DE BASE, pas une image source!
"""

import cv2
import json
import sys
from pathlib import Path


class IdentificationZoneSelector:
    """Sélecteur interactif de la zone d'identification."""
    
    def __init__(self, image_path: str, config_path: str = "ref/config.json"):
        self.image_path = image_path
        self.config_path = config_path
        self.image = None
        self.display_image = None
        self.zone = None
        self.drawing = False
        self.start_point = None
        
    def load_image(self):
        """Charge l'image de base."""
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            print(f"✗ Impossible de charger l'image: {self.image_path}")
            sys.exit(1)
        self.display_image = self.image.copy()
        print(f"✓ Image chargée: {self.image_path}")
        print(f"  Dimensions: {self.image.shape[1]}x{self.image.shape[0]} pixels")
        print()
        print("⚠ IMPORTANT: Assurez-vous que c'est bien une IMAGE DE BASE")
        print("  (pas une image source à analyser)")
        print()
        
    def mouse_callback(self, event, x, y, flags, param):
        """Gère les événements de la souris."""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp_image = self.image.copy()
                cv2.rectangle(temp_image, self.start_point, (x, y), (0, 255, 0), 2)
                # Afficher les coordonnées
                text = f"({self.start_point[0]}, {self.start_point[1]}) to ({x}, {y})"
                cv2.putText(temp_image, text, (self.start_point[0], self.start_point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                self.display_image = temp_image
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                end_point = (x, y)
                
                # Calculer les coordonnées
                x1 = min(self.start_point[0], end_point[0])
                y1 = min(self.start_point[1], end_point[1])
                x2 = max(self.start_point[0], end_point[0])
                y2 = max(self.start_point[1], end_point[1])
                
                width = x2 - x1
                height = y2 - y1
                
                if width > 10 and height > 10:
                    self.zone = {
                        'x': x1,
                        'y': y1,
                        'width': width,
                        'height': height
                    }
                    self.display_zone()
    
    def display_zone(self):
        """Affiche la zone sélectionnée."""
        if self.zone:
            temp_image = self.image.copy()
            x, y, w, h = self.zone['x'], self.zone['y'], self.zone['width'], self.zone['height']
            cv2.rectangle(temp_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Ajouter un label
            cv2.putText(temp_image, "Zone d'identification", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Afficher les dimensions
            text = f"{w}x{h} pixels"
            cv2.putText(temp_image, text, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            self.display_image = temp_image
    
    def run(self):
        """Lance l'interface interactive."""
        self.load_image()
        
        window_name = "Selection Zone d'Identification - Sur IMAGE DE BASE"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("="*60)
        print("SÉLECTION DE LA ZONE D'IDENTIFICATION")
        print("="*60)
        print("\nInstructions:")
        print("  1. Localisez 'Liste N° XX' en HAUT À DROITE de l'image")
        print("  2. Cliquez et glissez pour sélectionner cette zone")
        print("  3. La zone doit englober le texte 'Liste N° 01' (par exemple)")
        print("  4. Appuyez sur 's' pour sauvegarder")
        print("  5. Appuyez sur 'r' pour recommencer")
        print("  6. Appuyez sur 'q' pour quitter sans sauvegarder")
        print("="*60)
        print()
        
        while True:
            cv2.imshow(window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n✗ Annulation sans sauvegarde")
                break
                
            elif key == ord('s'):
                if self.zone:
                    self.save_zone()
                    print("\n✓ Zone sauvegardée, fermeture...")
                    break
                else:
                    print("\n⚠ Aucune zone sélectionnée!")
                
            elif key == ord('r'):
                self.zone = None
                self.display_image = self.image.copy()
                print("\n→ Zone réinitialisée, sélectionnez à nouveau")
        
        cv2.destroyAllWindows()
    
    def save_zone(self):
        """Sauvegarde la zone dans la configuration."""
        # Charger la config existante
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {
                "paths": {},
                "detection": {},
                "exclusion_zones": [],
                "processing": {}
            }
        
        # Mettre à jour la zone d'identification
        config['identification_zone'] = {
            'enabled': True,
            'x': self.zone['x'],
            'y': self.zone['y'],
            'width': self.zone['width'],
            'height': self.zone['height'],
            'description': "Zone du numéro de liste en haut à droite",
            'black_pixel_threshold_percent': 5.0
        }
        
        # Sauvegarder
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        print(f"\n✓ Zone d'identification sauvegardée dans {self.config_path}")
        print("\nCoordonnées enregistrées:")
        print(f"  • Position: ({self.zone['x']}, {self.zone['y']})")
        print(f"  • Dimensions: {self.zone['width']} x {self.zone['height']} pixels")
        print(f"  • Seuil: 5.0%")
        print()
        print("Vous pouvez maintenant tester avec:")
        print(f"  python test_identification_zone.py {self.image_path}")


def main():
    """Point d'entrée principal."""
    if len(sys.argv) < 2:
        print("\n⚠ IMPORTANT: Utilisez une IMAGE DE BASE, pas une image source!")
        print("\nUsage: python define_identification_zone.py <image_de_base>")
        print("\nExemple:")
        print("  python define_identification_zone.py images_base/vernier_SNL.png")
        print("  python define_identification_zone.py images_base/bulletin_LED_01.png")
        print("\nCe script permet de:")
        print("  • Sélectionner visuellement la zone 'Liste N° XX'")
        print("  • Sauvegarder les coordonnées dans config.json")
        print("  • Les coordonnées seront relatives à l'image de base")
        print("\n⚠ N'utilisez PAS une image source qui sera analysée,")
        print("  utilisez une IMAGE DE BASE (référence) !")
        print()
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"✗ Fichier non trouvé: {image_path}")
        sys.exit(1)
    
    # Vérifier que c'est bien dans images_base
    if "images_base" not in image_path and "base" not in image_path.lower():
        print("\n" + "="*60)
        print("⚠ ATTENTION")
        print("="*60)
        print("\nCe fichier ne semble pas être dans le dossier images_base/")
        print(f"Fichier: {image_path}")
        print()
        response = input("Êtes-vous sûr que c'est une image DE BASE (référence) ? (o/n): ")
        if response.lower() not in ['o', 'oui', 'y', 'yes']:
            print("\n✗ Opération annulée")
            print("  Utilisez une image de référence dans images_base/")
            sys.exit(1)
    
    selector = IdentificationZoneSelector(image_path)
    selector.run()


if __name__ == "__main__":
    main()