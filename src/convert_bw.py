#!/usr/bin/env python3
# src/convert_bw.py
"""
Module de conversion en noir et blanc pour la comparaison d'images.

MODES D'UTILISATION:

1. En tant que module (import):
   from src.convert_bw import convert_for_comparison, convert_and_resize_for_comparison

2. En ligne de commande - Mode référence (conversion batch):
   python convert_bw.py --mode ref --source ./ref_couleur --output ./ref_nb

   Convertit toutes les images d'un répertoire en N&B pour créer des références.
"""

import os
import sys
import cv2
import argparse
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Formats d'images supportés
SUPPORTED_FORMATS = [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"]


def convert_to_bw(img: np.ndarray, threshold: int = 127,
                  clahe_clip: float = 1.5, clahe_tile: int = 8) -> np.ndarray:
    """
    Convertit une image BGR en noir et blanc (binarisée).
    Les dimensions d'origine sont PRÉSERVÉES.
    
    Args:
        img: Image BGR (numpy array)
        threshold: Seuil de binarisation (0-255)
        clahe_clip: Facteur d'amélioration du contraste CLAHE
        clahe_tile: Taille des tuiles pour CLAHE
        
    Returns:
        Image binarisée BGR (numpy array), None en cas d'erreur
    """
    if img is None:
        logger.error("Image nulle reçue pour conversion N&B")
        return None
    
    try:
        original_height, original_width = img.shape[:2]
        
        # Convertir en niveaux de gris si nécessaire
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Amélioration du contraste avec CLAHE
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
        enhanced = clahe.apply(gray)
        
        # Binarisation
        _, binary = cv2.threshold(enhanced, threshold, 255, cv2.THRESH_BINARY)
        
        # Vérifier que les dimensions n'ont pas changé
        assert binary.shape == (original_height, original_width), \
            f"Dimensions modifiées: {binary.shape} != {(original_height, original_width)}"
        
        # Convertir en BGR pour compatibilité
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        logger.debug(f"Conversion N&B: {original_width}x{original_height}")
        return binary_bgr
    
    except Exception as e:
        logger.error(f"Erreur conversion N&B: {e}")
        return None


def convert_and_resize_for_comparison(src_img: np.ndarray, 
                                       ref_img: np.ndarray,
                                       config: dict = None) -> np.ndarray:
    """
    Convertit une image source et la redimensionne pour matcher la référence.
    
    IMPORTANT: Le redimensionnement est fait AVANT la binarisation pour
    éviter les artefacts d'interpolation sur une image binaire.
    
    Args:
        src_img: Image source BGR (potentiellement couleur)
        ref_img: Image de référence BGR (pour les dimensions)
        config: Configuration optionnelle
        
    Returns:
        Image source convertie et redimensionnée (BGR)
    """
    if src_img is None or ref_img is None:
        return None
    
    # Paramètres
    if config and "bw_conversion" in config:
        bw_config = config["bw_conversion"]
        threshold = bw_config.get("threshold", 127)
        clahe_clip = bw_config.get("clahe_clip", 1.5)
        clahe_tile = bw_config.get("clahe_tile", 8)
    else:
        threshold = 127
        clahe_clip = 1.5
        clahe_tile = 8
    
    ref_h, ref_w = ref_img.shape[:2]
    src_h, src_w = src_img.shape[:2]
    
    # 1. Redimensionner l'image SOURCE à la taille de la référence
    #    AVANT la conversion N&B (pour éviter artefacts d'interpolation)
    if (src_w, src_h) != (ref_w, ref_h):
        logger.debug(f"Redimensionnement: {src_w}x{src_h} -> {ref_w}x{ref_h}")
        src_resized = cv2.resize(src_img, (ref_w, ref_h), interpolation=cv2.INTER_AREA)
    else:
        src_resized = src_img
    
    # 2. Convertir en N&B
    result = convert_to_bw(src_resized, threshold, clahe_clip, clahe_tile)
    
    return result


def is_color_image(img: np.ndarray, saturation_threshold: float = 15.0) -> bool:
    """
    Détermine si une image est en couleur ou déjà en noir et blanc.
    
    Args:
        img: Image BGR
        saturation_threshold: Seuil de saturation moyenne
        
    Returns:
        True si l'image est en couleur
    """
    if img is None or len(img.shape) != 3:
        return False
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    mean_saturation = np.mean(saturation)
    
    is_color = mean_saturation > saturation_threshold
    logger.debug(f"Saturation moyenne: {mean_saturation:.2f}, couleur: {is_color}")
    
    return is_color


def convert_for_comparison(img: np.ndarray, config: dict = None) -> np.ndarray:
    """
    Convertit une image pour la comparaison avec un référentiel N&B.
    Si l'image est en couleur, elle est convertie en N&B.
    Si elle est déjà en N&B, elle est retournée telle quelle.
    
    Note: Cette fonction ne redimensionne PAS.
    """
    if img is None:
        return None
    
    if config and "bw_conversion" in config:
        bw_config = config["bw_conversion"]
        threshold = bw_config.get("threshold", 127)
        clahe_clip = bw_config.get("clahe_clip", 1.5)
        clahe_tile = bw_config.get("clahe_tile", 8)
        saturation_threshold = bw_config.get("saturation_threshold", 15.0)
    else:
        threshold = 127
        clahe_clip = 1.5
        clahe_tile = 8
        saturation_threshold = 15.0
    
    if is_color_image(img, saturation_threshold):
        logger.info("Image couleur détectée - conversion N&B")
        return convert_to_bw(img, threshold, clahe_clip, clahe_tile)
    else:
        logger.debug("Image déjà N&B")
        return img


# =============================================================================
# MODE BATCH - Conversion d'un répertoire de références
# =============================================================================

def convert_reference_directory(source_dir: str, output_dir: str,
                                 threshold: int = 127,
                                 clahe_clip: float = 1.5,
                                 clahe_tile: int = 8) -> dict:
    """
    Convertit toutes les images d'un répertoire en N&B.
    
    Args:
        source_dir: Répertoire contenant les images sources
        output_dir: Répertoire de destination
        threshold: Seuil de binarisation (0-255)
        clahe_clip: Facteur CLAHE
        clahe_tile: Taille des tuiles CLAHE
        
    Returns:
        Dictionnaire avec statistiques {success, failed, total, details}
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Vérifications
    if not source_path.exists():
        logger.error(f"Répertoire source introuvable: {source_dir}")
        return {"success": 0, "failed": 0, "total": 0, "error": "Source introuvable"}
    
    if not source_path.is_dir():
        logger.error(f"Le chemin source n'est pas un répertoire: {source_dir}")
        return {"success": 0, "failed": 0, "total": 0, "error": "Source n'est pas un répertoire"}
    
    # Créer le répertoire de destination
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collecter les images
    image_files = []
    for ext in SUPPORTED_FORMATS:
        image_files.extend(source_path.glob(f"*{ext}"))
        image_files.extend(source_path.glob(f"*{ext.upper()}"))
    
    image_files = sorted(set(image_files))
    
    if not image_files:
        logger.warning(f"Aucune image trouvée dans {source_dir}")
        return {"success": 0, "failed": 0, "total": 0, "details": []}
    
    # Traitement
    success_count = 0
    failed_count = 0
    details = []
    
    logger.info(f"Conversion de {len(image_files)} image(s)...")
    
    for img_path in image_files:
        # Charger l'image
        img = cv2.imread(str(img_path))
        
        if img is None:
            logger.error(f"Impossible de lire: {img_path.name}")
            failed_count += 1
            details.append({
                "file": img_path.name,
                "status": "error",
                "message": "Lecture impossible"
            })
            continue
        
        # Dimensions originales
        orig_h, orig_w = img.shape[:2]
        
        # Convertir
        converted = convert_to_bw(img, threshold, clahe_clip, clahe_tile)
        
        if converted is None:
            logger.error(f"Échec de conversion: {img_path.name}")
            failed_count += 1
            details.append({
                "file": img_path.name,
                "status": "error",
                "message": "Conversion échouée"
            })
            continue
        
        # Vérifier les dimensions
        conv_h, conv_w = converted.shape[:2]
        if (conv_w, conv_h) != (orig_w, orig_h):
            logger.warning(f"Dimensions modifiées pour {img_path.name}")
        
        # Nom de sortie (toujours PNG)
        output_filename = img_path.stem + ".png"
        output_file = output_path / output_filename
        
        # Sauvegarder en PNG sans compression
        success = cv2.imwrite(str(output_file), converted, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        if success:
            success_count += 1
            logger.info(f"✓ {img_path.name} → {output_filename} ({orig_w}x{orig_h})")
            details.append({
                "file": img_path.name,
                "output": output_filename,
                "status": "success",
                "dimensions": f"{orig_w}x{orig_h}"
            })
        else:
            failed_count += 1
            logger.error(f"✗ Échec sauvegarde: {output_filename}")
            details.append({
                "file": img_path.name,
                "status": "error",
                "message": "Sauvegarde échouée"
            })
    
    return {
        "success": success_count,
        "failed": failed_count,
        "total": len(image_files),
        "details": details
    }


def main():
    """Point d'entrée principal pour l'utilisation en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Conversion d'images en noir et blanc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Convertir un répertoire de références
  python convert_bw.py --mode ref --source ./ref_couleur --output ./ref_nb

  # Avec paramètres personnalisés
  python convert_bw.py --mode ref -s ./source -o ./output --threshold 130
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["ref"],
        required=True,
        help="Mode d'exécution: 'ref' pour convertir un répertoire de références"
    )
    
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Répertoire source contenant les images à convertir"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Répertoire de destination pour les images converties"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=127,
        help="Seuil de binarisation 0-255 (défaut: 127)"
    )
    
    parser.add_argument(
        "--clahe-clip",
        type=float,
        default=1.5,
        help="Facteur CLAHE (défaut: 1.5)"
    )
    
    parser.add_argument(
        "--clahe-tile",
        type=int,
        default=8,
        help="Taille tuile CLAHE (défaut: 8)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Afficher les détails de traitement"
    )
    
    args = parser.parse_args()
    
    # Configuration du logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(message)s"
    )
    
    # Exécution selon le mode
    if args.mode == "ref":
        print()
        print("=" * 60)
        print("CONVERSION EN NOIR ET BLANC - MODE RÉFÉRENCE")
        print("=" * 60)
        print(f"Source      : {args.source}")
        print(f"Destination : {args.output}")
        print(f"Threshold   : {args.threshold}")
        print(f"CLAHE       : clip={args.clahe_clip}, tile={args.clahe_tile}")
        print("=" * 60)
        print()
        
        # Conversion
        stats = convert_reference_directory(
            source_dir=args.source,
            output_dir=args.output,
            threshold=args.threshold,
            clahe_clip=args.clahe_clip,
            clahe_tile=args.clahe_tile
        )
        
        # Résumé
        print()
        print("=" * 60)
        print("RÉSUMÉ")
        print("=" * 60)
        print(f"  Total     : {stats['total']} image(s)")
        print(f"  ✓ Succès  : {stats['success']} image(s)")
        
        if stats['failed'] > 0:
            print(f"  ✗ Échecs  : {stats['failed']} image(s)")
        
        if stats['success'] == stats['total'] and stats['total'] > 0:
            print()
            print("✓ Toutes les images ont été converties avec succès!")
        
        print("=" * 60)
        print()
        
        # Code de retour
        sys.exit(0 if stats['failed'] == 0 else 1)


if __name__ == "__main__":
    main()