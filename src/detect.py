#!/usr/bin/env python3
"""
Script de détection d'annotations sur des formulaires scannés.
Compare des images scannées avec des images de référence vierges.
Version 2: Avec identification automatique du type de bulletin (avec/sans numéro de liste)
"""

import cv2
import numpy as np
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import shutil


class FormAnnotationDetector:
    """Détecteur d'annotations sur formulaires scannés."""
    
    def __init__(self, config_path: str = "ref/config.json"):
        """
        Initialise le détecteur avec la configuration.
        
        Args:
            config_path: Chemin vers le fichier de configuration JSON
        """
        self.config = self.load_config(config_path)
        self.setup_directories()
        
    def load_config(self, config_path: str) -> dict:
        """Charge la configuration depuis le fichier JSON."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✓ Configuration chargée depuis {config_path}")
            return config
        except FileNotFoundError:
            print(f"⚠ Fichier de configuration non trouvé. Création de {config_path}")
            default_config = self.create_default_config()
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            return default_config
    
    def create_default_config(self) -> dict:
        """Crée une configuration par défaut."""
        return {
            "paths": {
                "base_images_dir": "./images_base",
                "source_images_dir": "./images_source",
                "export_dir": "./export",
                "csv_output": "./export/resultats_analyse.csv"
            },
            "detection": {
                "difference_threshold_percent": 2.0,
                "corner_detection_threshold": 50,
                "alignment_tolerance": 0.5
            },
            "identification_zone": {
                "enabled": True,
                "x": 550,
                "y": 620,
                "width": 130,
                "height": 50,
                "description": "Zone du numéro de liste en haut à droite",
                "black_pixel_threshold_percent": 5.0
            },
            "exclusion_zones": [
                {
                    "name": "numero_document",
                    "x": 50,
                    "y": 1100,
                    "width": 400,
                    "height": 50,
                    "description": "Numéro de document en bas"
                }
            ],
            "processing": {
                "rotation_range": 180,
                "rotation_step": 0.5,
                "corner_template_size": 100,
                "binarization_threshold": 127
            }
        }
    
    def setup_directories(self):
        """Crée les répertoires nécessaires s'ils n'existent pas."""
        paths = self.config['paths']
        
        # Répertoires principaux
        Path(paths['base_images_dir']).mkdir(parents=True, exist_ok=True)
        Path(paths['source_images_dir']).mkdir(parents=True, exist_ok=True)
        Path(paths['export_dir']).mkdir(parents=True, exist_ok=True)
        
        # Sous-répertoires d'export
        Path(paths['export_dir'], 'modifie').mkdir(parents=True, exist_ok=True)
        Path(paths['export_dir'], 'pas-modifie').mkdir(parents=True, exist_ok=True)
        
        print("✓ Répertoires initialisés")
    
    def detect_bulletin_type(self, aligned_image: np.ndarray) -> str:
        """
        Détecte le type de bulletin en analysant la zone d'identification.
        
        Args:
            aligned_image: Image alignée en niveaux de gris
            
        Returns:
            "AVEC_NUMERO" si un numéro de liste est détecté, "SNL" sinon
        """
        if not self.config.get('identification_zone', {}).get('enabled', False):
            return "INCONNU"
        
        id_zone = self.config['identification_zone']
        x, y, w, h = id_zone['x'], id_zone['y'], id_zone['width'], id_zone['height']
        threshold_percent = id_zone.get('black_pixel_threshold_percent', 5.0)
        
        # Vérifier que la zone est dans les limites de l'image
        img_h, img_w = aligned_image.shape[:2]
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            print(f"  ⚠ Zone d'identification hors limites")
            return "INCONNU"
        
        # Extraire la zone
        zone = aligned_image[y:y+h, x:x+w]
        
        # Binariser
        _, binary = cv2.threshold(
            zone, 
            self.config['processing']['binarization_threshold'], 
            255, 
            cv2.THRESH_BINARY
        )
        
        # Inverser (noir = 255)
        inverted = cv2.bitwise_not(binary)
        
        # Calculer le pourcentage de pixels noirs
        total_pixels = zone.size
        black_pixels = np.count_nonzero(inverted)
        black_percent = (black_pixels / total_pixels) * 100
        
        # Décision
        if black_percent >= threshold_percent:
            return "AVEC_NUMERO"
        else:
            return "SNL"
    
    def detect_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Détecte les 4 coins du formulaire (marqueurs en L).
        Si les marqueurs ne sont pas trouvés, utilise les coins de l'image.
        
        Args:
            image: Image en niveaux de gris
            
        Returns:
            Array de 4 points (coins) ou None si non trouvés
        """
        # Binarisation
        _, binary = cv2.threshold(
            image, 
            self.config['processing']['binarization_threshold'], 
            255, 
            cv2.THRESH_BINARY_INV
        )
        
        # Détection des contours
        contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Recherche des marqueurs en forme de L dans les coins
        h, w = image.shape
        corners = []
        
        # Définir les zones pour chaque coin
        corner_regions = [
            (0, 0, w//4, h//4),           # Haut gauche
            (3*w//4, 0, w, h//4),         # Haut droit
            (0, 3*h//4, w//4, h),         # Bas gauche
            (3*w//4, 3*h//4, w, h)        # Bas droit
        ]
        
        for region in corner_regions:
            x1, y1, x2, y2 = region
            region_contours = [
                cnt for cnt in contours 
                if self._contour_in_region(cnt, region)
            ]
            
            if region_contours:
                # Trouver le plus grand contour dans cette région
                largest = max(region_contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    corners.append([cx, cy])
        
        if len(corners) == 4:
            # Ordonner les coins: haut-gauche, haut-droit, bas-droit, bas-gauche
            corners = np.array(corners, dtype=np.float32)
            corners = self._order_points(corners)
            return corners
        
        # FALLBACK: Si pas de marqueurs en L, utiliser les bords de l'image avec marge
        print("  ℹ Utilisation des bords de l'image (pas de marqueurs L détectés)")
        margin = 50  # Marge de 50 pixels depuis les bords
        corners = np.array([
            [margin, margin],                    # Haut-gauche
            [w - margin, margin],                # Haut-droit
            [w - margin, h - margin],            # Bas-droit
            [margin, h - margin]                 # Bas-gauche
        ], dtype=np.float32)
        
        return corners
    
    def _contour_in_region(self, contour: np.ndarray, region: Tuple[int, int, int, int]) -> bool:
        """Vérifie si un contour est dans une région donnée."""
        x1, y1, x2, y2 = region
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return x1 <= cx <= x2 and y1 <= cy <= y2
        return False
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Ordonne les points dans l'ordre: haut-gauche, haut-droit, bas-droit, bas-gauche.
        """
        # Initialiser les coordonnées ordonnées
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Somme et différence des coordonnées
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Haut-gauche aura la plus petite somme
        # Bas-droit aura la plus grande somme
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Haut-droit aura la plus petite différence
        # Bas-gauche aura la plus grande différence
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def align_images(self, source_img: np.ndarray, base_img: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Aligne l'image source avec l'image de base en utilisant les coins.
        
        Args:
            source_img: Image à aligner
            base_img: Image de référence
            
        Returns:
            Tuple (image alignée, score de confiance) ou (None, 0.0) si échec
        """
        # Convertir en niveaux de gris si nécessaire
        if len(source_img.shape) == 3:
            source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
        else:
            source_gray = source_img.copy()
            
        if len(base_img.shape) == 3:
            base_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        else:
            base_gray = base_img.copy()
        
        # Détecter les coins
        source_corners = self.detect_corners(source_gray)
        base_corners = self.detect_corners(base_gray)
        
        if source_corners is None or base_corners is None:
            print("  ⚠ Impossible de détecter les coins")
            return None, 0.0
        
        # Calculer la transformation perspective
        matrix = cv2.getPerspectiveTransform(source_corners, base_corners)
        
        # Appliquer la transformation
        h, w = base_gray.shape
        aligned = cv2.warpPerspective(source_gray, matrix, (w, h))
        
        # Calculer le score de confiance basé sur la distance entre les coins
        distances = np.linalg.norm(source_corners - base_corners, axis=1)
        avg_distance = np.mean(distances)
        max_distance = np.sqrt(w**2 + h**2)  # Distance diagonale maximale
        confidence = max(0, 1 - (avg_distance / max_distance))
        
        return aligned, confidence
    
    def apply_exclusion_zones(self, image: np.ndarray) -> np.ndarray:
        """
        Applique un masque blanc sur les zones d'exclusion.
        
        Args:
            image: Image à masquer
            
        Returns:
            Image avec zones d'exclusion masquées
        """
        masked = image.copy()
        
        for zone in self.config['exclusion_zones']:
            x, y, w, h = zone['x'], zone['y'], zone['width'], zone['height']
            # Vérifier que la zone est dans les limites de l'image
            img_h, img_w = masked.shape[:2]
            if x >= 0 and y >= 0 and x + w <= img_w and y + h <= img_h:
                cv2.rectangle(masked, (x, y), (x + w, y + h), 255, -1)
            else:
                print(f"  ⚠ Zone d'exclusion '{zone['name']}' hors limites, ignorée")
        
        return masked
    
    def calculate_difference(self, source_img: np.ndarray, base_img: np.ndarray) -> float:
        """
        Calcule le pourcentage de pixels plus noirs dans l'image source.
        
        Args:
            source_img: Image source (potentiellement annotée)
            base_img: Image de référence (vierge)
            
        Returns:
            Pourcentage de différence (pixels noirs ajoutés)
        """
        # Appliquer les zones d'exclusion
        source_masked = self.apply_exclusion_zones(source_img)
        base_masked = self.apply_exclusion_zones(base_img)
        
        # Binarisation
        threshold = self.config['processing']['binarization_threshold']
        _, source_bin = cv2.threshold(source_masked, threshold, 255, cv2.THRESH_BINARY)
        _, base_bin = cv2.threshold(base_masked, threshold, 255, cv2.THRESH_BINARY)
        
        # Inverser pour que le noir soit 255 et le blanc 0
        source_inv = cv2.bitwise_not(source_bin)
        base_inv = cv2.bitwise_not(base_bin)
        
        # Calculer la différence (pixels noirs en plus dans source)
        diff = cv2.subtract(source_inv, base_inv)
        
        # Compter les pixels noirs ajoutés
        total_pixels = diff.size
        added_black_pixels = np.count_nonzero(diff)
        
        # Calculer le pourcentage
        percentage = (added_black_pixels / total_pixels) * 100
        
        return percentage
    
    def find_best_match(self, source_path: Path) -> Tuple[Optional[Path], float, float, str]:
        """
        Trouve la meilleure image de base correspondante.
        
        Args:
            source_path: Chemin de l'image source
            
        Returns:
            Tuple (chemin de la meilleure base, différence %, score de confiance, type de bulletin)
        """
        source_img = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        if source_img is None:
            print(f"  ✗ Impossible de charger {source_path}")
            return None, 0.0, 0.0, "INCONNU"
        
        # Aligner d'abord avec une image de référence pour détecter le type
        base_dir = Path(self.config['paths']['base_images_dir'])
        base_images = list(base_dir.glob('*.png')) + list(base_dir.glob('*.jpg')) + list(base_dir.glob('*.jpeg'))
        
        if not base_images:
            print(f"  ✗ Aucune image de base trouvée dans {base_dir}")
            return None, 0.0, 0.0, "INCONNU"
        
        # Utiliser la première image pour un alignement initial et détecter le type
        first_base = cv2.imread(str(base_images[0]), cv2.IMREAD_GRAYSCALE)
        aligned_temp, _ = self.align_images(source_img, first_base)
        
        if aligned_temp is None:
            print(f"  ✗ Impossible d'aligner pour détecter le type")
            return None, 0.0, 0.0, "INCONNU"
        
        # Détecter le type de bulletin
        bulletin_type = self.detect_bulletin_type(aligned_temp)
        print(f"  → Type de bulletin détecté: {bulletin_type}")
        
        # Filtrer les images de base selon le type
        # Convention de nommage: les images SNL doivent contenir "SNL" dans leur nom
        if bulletin_type == "SNL":
            base_images = [img for img in base_images if "SNL" in img.stem.upper() or "snl" in img.stem]
        else:
            base_images = [img for img in base_images if "SNL" not in img.stem.upper() and "snl" not in img.stem]
        
        if not base_images:
            print(f"  ⚠ Aucune image de base du type {bulletin_type} trouvée")
            # Fallback: utiliser toutes les images
            base_images = list(base_dir.glob('*.png')) + list(base_dir.glob('*.jpg')) + list(base_dir.glob('*.jpeg'))
        
        best_match = None
        best_difference = float('inf')
        best_confidence = 0.0
        
        print(f"  → Comparaison avec {len(base_images)} image(s) de base du type {bulletin_type}...")
        
        for base_path in base_images:
            base_img = cv2.imread(str(base_path), cv2.IMREAD_GRAYSCALE)
            if base_img is None:
                continue
            
            # Aligner les images
            aligned, confidence = self.align_images(source_img, base_img)
            
            if aligned is None:
                continue
            
            # Calculer la différence
            difference = self.calculate_difference(aligned, base_img)
            
            print(f"    • {base_path.name}: {difference:.2f}% différence, confiance: {confidence:.3f}")
            
            # Garder la meilleure correspondance (différence minimale)
            if difference < best_difference:
                best_difference = difference
                best_match = base_path
                best_confidence = confidence
        
        return best_match, best_difference, best_confidence, bulletin_type
    
    def process_images(self):
        """Traite toutes les images du répertoire source."""
        source_dir = Path(self.config['paths']['source_images_dir'])
        export_dir = Path(self.config['paths']['export_dir'])
        csv_path = Path(self.config['paths']['csv_output'])
        
        # Collecter toutes les images source
        source_images = (
            list(source_dir.glob('*.png')) + 
            list(source_dir.glob('*.jpg')) + 
            list(source_dir.glob('*.jpeg'))
        )
        
        if not source_images:
            print(f"✗ Aucune image trouvée dans {source_dir}")
            return
        
        print(f"\n{'='*70}")
        print(f"Traitement de {len(source_images)} image(s)")
        print(f"{'='*70}\n")
        
        # Préparer le fichier CSV
        results = []
        threshold = self.config['detection']['difference_threshold_percent']
        print(f"ℹ Seuil de détection configuré: {threshold}%\n")
        
        for i, source_path in enumerate(source_images, 1):
            print(f"[{i}/{len(source_images)}] Analyse de {source_path.name}")
            
            # Trouver la meilleure correspondance
            best_base, difference, confidence, bulletin_type = self.find_best_match(source_path)
            
            if best_base is None:
                print(f"  ✗ Échec de l'analyse\n")
                results.append({
                    'fichier_source': source_path.name,
                    'type_bulletin': 'ERREUR',
                    'fichier_reference': 'N/A',
                    'pourcentage_difference': 'N/A',
                    'score_confiance': 'N/A',
                    'statut': 'ERREUR'
                })
                continue
            
            # Déterminer si modifié
            is_modified = difference >= threshold
            status = 'MODIFIE' if is_modified else 'PAS_MODIFIE'
            
            # Copier l'image dans le bon répertoire
            if is_modified:
                dest_dir = export_dir / 'modifie'
            else:
                dest_dir = export_dir / 'pas-modifie'
            
            shutil.copy2(source_path, dest_dir / source_path.name)
            
            # Enregistrer les résultats
            results.append({
                'fichier_source': source_path.name,
                'type_bulletin': bulletin_type,
                'fichier_reference': best_base.name,
                'pourcentage_difference': f"{difference:.3f}",
                'score_confiance': f"{confidence:.3f}",
                'statut': status
            })
            
            print(f"  ✓ Type: {bulletin_type}")
            print(f"  ✓ Résultat: {status} ({difference:.2f}% vs seuil {threshold}%)")
            print(f"  ✓ Copié dans {dest_dir.name}/\n")
        
        # Écrire le CSV
        self.write_csv(results, csv_path)
        
        print(f"{'='*70}")
        print(f"✓ Traitement terminé!")
        print(f"✓ Résultats exportés dans {csv_path}")
        print(f"{'='*70}\n")
    
    def write_csv(self, results: List[Dict], csv_path: Path):
        """Écrit les résultats dans un fichier CSV."""
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'fichier_source',
                'type_bulletin',
                'fichier_reference', 
                'pourcentage_difference',
                'score_confiance',
                'statut'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"✓ CSV créé: {csv_path}")


def main():
    """Point d'entrée principal du script."""
    print("\n" + "="*70)
    print("DÉTECTEUR D'ANNOTATIONS SUR FORMULAIRES SCANNÉS")
    print("Version 2 - Avec identification automatique du type de bulletin")
    print("="*70 + "\n")
    
    # Initialiser le détecteur
    detector = FormAnnotationDetector("ref/config.json")
    
    # Vérifier que les répertoires nécessaires existent et contiennent des fichiers
    base_dir = Path(detector.config['paths']['base_images_dir'])
    source_dir = Path(detector.config['paths']['source_images_dir'])
    
    base_images = list(base_dir.glob('*.png')) + list(base_dir.glob('*.jpg'))
    source_images = list(source_dir.glob('*.png')) + list(source_dir.glob('*.jpg'))
    
    if not base_images:
        print(f"⚠ ATTENTION: Aucune image de base trouvée dans '{base_dir}'")
        print(f"  Veuillez placer vos images de référence vierges dans ce répertoire.")
        print(f"  Convention de nommage:")
        print(f"    - Images SNL (Sans Nom de Liste): inclure 'SNL' dans le nom")
        print(f"    - Images avec numéro: ne pas inclure 'SNL' dans le nom\n")
        return
    
    if not source_images:
        print(f"⚠ ATTENTION: Aucune image source trouvée dans '{source_dir}'")
        print(f"  Veuillez placer vos images à analyser dans ce répertoire.\n")
        return
    
    print(f"✓ {len(base_images)} image(s) de base trouvée(s)")
    
    # Compter les images SNL et avec numéro
    snl_count = sum(1 for img in base_images if "SNL" in img.stem.upper() or "snl" in img.stem)
    avec_numero_count = len(base_images) - snl_count
    
    print(f"  • {snl_count} image(s) SNL (Sans Nom de Liste)")
    print(f"  • {avec_numero_count} image(s) avec numéro de liste")
    print(f"✓ {len(source_images)} image(s) source trouvée(s)\n")
    
    # Traiter les images
    detector.process_images()


if __name__ == "__main__":
    main()