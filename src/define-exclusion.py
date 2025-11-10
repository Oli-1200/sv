#!/usr/bin/env python3
"""
Utilitaire pour définir interactivement les zones d'exclusion.
Permet de sélectionner des rectangles sur une image de référence.
"""

import cv2
import json
import sys
from pathlib import Path


class ExclusionZoneSelector:
    """Sélecteur interactif de zones d'exclusion."""
    
    def __init__(self, image_path: str, config_path: str = "ref/config.json"):
        self.image_path = image_path
        self.config_path = config_path
        self.image = None
        self.display_image = None
        self.zones = []
        self.current_rect = None
        self.drawing = False
        self.start_point = None
        
    def load_image(self):
        """Charge l'image de référence."""
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            print(f"✗ Impossible de charger l'image: {self.image_path}")
            sys.exit(1)
        self.display_image = self.image.copy()
        print(f"✓ Image chargée: {self.image_path}")
        print(f"  Dimensions: {self.image.shape[1]}x{self.image.shape[0]} pixels")
        
    def mouse_callback(self, event, x, y, flags, param):
        """Gère les événements de la souris."""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Début de la sélection
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # Mise à jour du rectangle pendant le dessin
            if self.drawing:
                temp_image = self.image.copy()
                
                # Dessiner les zones existantes
                for zone in self.zones:
                    cv2.rectangle(
                        temp_image,
                        (zone['x'], zone['y']),
                        (zone['x'] + zone['width'], zone['y'] + zone['height']),
                        (0, 255, 0),
                        2
                    )
                    # Ajouter le nom de la zone
                    cv2.putText(
                        temp_image,
                        zone['name'],
                        (zone['x'], zone['y'] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )
                
                # Dessiner le rectangle en cours
                cv2.rectangle(
                    temp_image,
                    self.start_point,
                    (x, y),
                    (0, 0, 255),
                    2
                )
                
                self.display_image = temp_image
                
        elif event == cv2.EVENT_LBUTTONUP:
            # Fin de la sélection
            if self.drawing:
                self.drawing = False
                end_point = (x, y)
                
                # Calculer les coordonnées du rectangle
                x1 = min(self.start_point[0], end_point[0])
                y1 = min(self.start_point[1], end_point[1])
                x2 = max(self.start_point[0], end_point[0])
                y2 = max(self.start_point[1], end_point[1])
                
                width = x2 - x1
                height = y2 - y1
                
                # Vérifier que le rectangle a une taille minimale
                if width > 10 and height > 10:
                    # Demander le nom de la zone
                    self.add_zone(x1, y1, width, height)
                
                # Redessiner l'image
                self.redraw()
    
    def add_zone(self, x: int, y: int, width: int, height: int):
        """Ajoute une zone d'exclusion."""
        print(f"\n{'='*60}")
        print(f"Zone sélectionnée: x={x}, y={y}, largeur={width}, hauteur={height}")
        name = input("Nom de cette zone d'exclusion: ").strip()
        
        if not name:
            name = f"zone_{len(self.zones) + 1}"
        
        description = input("Description (optionnel): ").strip()
        if not description:
            description = f"Zone d'exclusion {len(self.zones) + 1}"
        
        zone = {
            'name': name,
            'x': int(x),
            'y': int(y),
            'width': int(width),
            'height': int(height),
            'description': description
        }
        
        self.zones.append(zone)
        print(f"✓ Zone '{name}' ajoutée")
        print(f"{'='*60}\n")
    
    def redraw(self):
        """Redessine l'image avec toutes les zones."""
        self.display_image = self.image.copy()
        
        for zone in self.zones:
            cv2.rectangle(
                self.display_image,
                (zone['x'], zone['y']),
                (zone['x'] + zone['width'], zone['y'] + zone['height']),
                (0, 255, 0),
                2
            )
            
            # Ajouter le nom de la zone
            cv2.putText(
                self.display_image,
                zone['name'],
                (zone['x'], zone['y'] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
    
    def run(self):
        """Lance l'interface interactive."""
        self.load_image()
        
        window_name = "Sélection des zones d'exclusion"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\n" + "="*60)
        print("SÉLECTION DES ZONES D'EXCLUSION")
        print("="*60)
        print("\nInstructions:")
        print("  • Cliquez et glissez pour sélectionner une zone")
        print("  • Relâchez pour valider la zone")
        print("  • Appuyez sur 'r' pour supprimer la dernière zone")
        print("  • Appuyez sur 's' pour sauvegarder et quitter")
        print("  • Appuyez sur 'q' pour quitter sans sauvegarder")
        print("="*60 + "\n")
        
        while True:
            cv2.imshow(window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quitter sans sauvegarder
                print("\n✗ Annulation sans sauvegarde")
                break
                
            elif key == ord('s'):
                # Sauvegarder et quitter
                self.save_zones()
                print("\n✓ Zones sauvegardées, fermeture...")
                break
                
            elif key == ord('r'):
                # Supprimer la dernière zone
                if self.zones:
                    removed = self.zones.pop()
                    print(f"✓ Zone '{removed['name']}' supprimée")
                    self.redraw()
                else:
                    print("⚠ Aucune zone à supprimer")
        
        cv2.destroyAllWindows()
    
    def save_zones(self):
        """Sauvegarde les zones dans le fichier de configuration."""
        if not self.zones:
            print("⚠ Aucune zone à sauvegarder")
            return
        
        # Charger la configuration existante
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"⚠ Fichier {self.config_path} non trouvé, création d'une nouvelle config")
            config = {
                "paths": {},
                "detection": {},
                "processing": {},
                "exclusion_zones": []
            }
        
        # Demander si on remplace ou on ajoute
        if config.get('exclusion_zones'):
            print(f"\n{len(config['exclusion_zones'])} zone(s) existante(s) dans la config")
            choice = input("Remplacer (r) ou Ajouter (a)? [r/a]: ").strip().lower()
            
            if choice == 'a':
                config['exclusion_zones'].extend(self.zones)
            else:
                config['exclusion_zones'] = self.zones
        else:
            config['exclusion_zones'] = self.zones
        
        # Sauvegarder
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        print(f"\n✓ {len(self.zones)} zone(s) sauvegardée(s) dans {self.config_path}")
        
        # Afficher un résumé
        print("\nZones sauvegardées:")
        for i, zone in enumerate(self.zones, 1):
            print(f"  {i}. {zone['name']}: ({zone['x']}, {zone['y']}) "
                  f"{zone['width']}x{zone['height']} - {zone['description']}")


def main():
    """Point d'entrée principal."""
    if len(sys.argv) < 2:
        print("Usage: python define_exclusions.py <chemin_image_reference>")
        print("\nExemple:")
        print("  python define_exclusions.py images_base/vernier.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"✗ Fichier non trouvé: {image_path}")
        sys.exit(1)
    
    selector = ExclusionZoneSelector(image_path)
    selector.run()


if __name__ == "__main__":
    main()