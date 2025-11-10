#!/usr/bin/env python3
"""
Script de conversion d'images en noir et blanc avec fond blanc.
Optimisé pour capturer tous les textes colorés (bleu, rouge, noir) sur fond blanc.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# ============================================================================
# PARAMÈTRES CONFIGURABLES
# ============================================================================

# Répertoires source et destination
SOURCE_DIR = "tiffNB/tiff"
DESTINATION_DIR = "ref/bulletins"

# Formats d'images à traiter (en minuscules) ".jpg", ".jpeg", ".png", ".tiff", ".tif"
IMAGE_FORMATS = [".tif"]

# Paramètres de traitement
CLAHE_CLIP_LIMIT = 1.5  # Amélioration du contraste (1.0 = normal, >1.0 = plus de contraste)
CLAHE_TILE_SIZE = 8     # Taille des tuiles pour CLAHE (8x8 recommandé)
THRESHOLD_VALUE = 180   # Seuil de binarisation (0-255)

# Qualité PNG (0-9, où 0 = pas de compression, 9 = compression maximale)
PNG_COMPRESSION = 0

# ============================================================================
# FONCTIONS
# ============================================================================

def create_destination_dir(dest_dir):
    """Crée le répertoire de destination s'il n'existe pas."""
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    return dest_path


def get_image_files(source_dir, formats):
    """Récupère tous les fichiers images du répertoire source."""
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"❌ Erreur : Le répertoire source '{source_dir}' n'existe pas.")
        sys.exit(1)
    
    image_files = []
    for ext in formats:
        # Cherche les fichiers avec extension en minuscules et majuscules
        image_files.extend(source_path.glob(f"*{ext}"))
        image_files.extend(source_path.glob(f"*{ext.upper()}"))
    
    return sorted(set(image_files))


def convert_image(input_path, output_path, clahe_clip, clahe_tile, threshold, png_compression):
    """
    Convertit une image en noir et blanc avec fond blanc et textes noirs.
    Capture tous les textes colorés (bleu, rouge, noir, etc.).
    
    Args:
        input_path: Chemin de l'image source
        output_path: Chemin de l'image de destination
        clahe_clip: Facteur d'amélioration du contraste CLAHE
        clahe_tile: Taille des tuiles pour CLAHE
        threshold: Valeur de seuil pour la binarisation
        png_compression: Niveau de compression PNG (0-9)
    """
    try:
        # Lire l'image
        img = cv2.imread(str(input_path))
        
        if img is None:
            raise ValueError(f"Impossible de lire l'image")
        
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Amélioration du contraste avec CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
        enhanced = clahe.apply(gray)
        
        # Binarisation avec seuil fixe
        _, binary = cv2.threshold(enhanced, threshold, 255, cv2.THRESH_BINARY)
        
        # Sauvegarder en PNG
        cv2.imwrite(str(output_path), binary, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
        
        return True
    
    except Exception as e:
        print(f"\n❌ Erreur lors du traitement de {input_path.name}: {e}")
        return False


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

def main():
    print("=" * 70)
    print("CONVERSION D'IMAGES EN NOIR ET BLANC (FOND BLANC + TEXTE NOIR)")
    print("=" * 70)
    print()
    
    # Afficher les paramètres
    print("📋 Paramètres :")
    print(f"   - Répertoire source      : {SOURCE_DIR}")
    print(f"   - Répertoire destination : {DESTINATION_DIR}")
    print(f"   - Formats acceptés       : {', '.join(IMAGE_FORMATS)}")
    print(f"   - CLAHE Clip Limit       : {CLAHE_CLIP_LIMIT}")
    print(f"   - CLAHE Tile Size        : {CLAHE_TILE_SIZE}x{CLAHE_TILE_SIZE}")
    print(f"   - Seuil de binarisation  : {THRESHOLD_VALUE}")
    print(f"   - Compression PNG        : {PNG_COMPRESSION}")
    print()
    
    # Créer le répertoire de destination
    dest_path = create_destination_dir(DESTINATION_DIR)
    print(f"✅ Répertoire de destination créé/vérifié : {DESTINATION_DIR}")
    print()
    
    # Récupérer la liste des images
    image_files = get_image_files(SOURCE_DIR, IMAGE_FORMATS)
    
    if not image_files:
        print(f"⚠️  Aucune image trouvée dans '{SOURCE_DIR}' avec les formats : {', '.join(IMAGE_FORMATS)}")
        return
    
    print(f"📊 Nombre d'images trouvées en entrée : {len(image_files)}")
    print()
    
    # Traiter les images avec barre de progression
    success_count = 0
    print("🔄 Traitement en cours...")
    
    for img_path in tqdm(image_files, desc="Conversion", unit="image"):
        # Créer le nom du fichier de sortie (même nom, extension .png)
        output_filename = img_path.stem + ".png"
        output_path = dest_path / output_filename
        
        # Convertir l'image
        if convert_image(img_path, output_path, CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE, 
                        THRESHOLD_VALUE, PNG_COMPRESSION):
            success_count += 1
    
    print()
    print("=" * 70)
    print("📊 RÉSUMÉ")
    print("=" * 70)
    print(f"   - Images en entrée  : {len(image_files)}")
    print(f"   - Images en sortie  : {success_count}")
    
    if success_count < len(image_files):
        print(f"   - Échecs            : {len(image_files) - success_count}")
    
    print()
    print("✅ Traitement terminé !")


if __name__ == "__main__":
    main()