#  License Plate Deblurring

> Restauration d'images de plaques d'immatriculation floues via un pipeline hybride traitement du signal + deep learning.

---

##  Présentation

Dans la vidéosurveillance, le contrôle d'accès ou l'analyse forensique, un léger mouvement du véhicule ou de la caméra suffit à rendre la lecture d'une plaque d'immatriculation difficile. Ce projet implémente un **pipeline complet de défloutage** : génération de données synthétiques, estimation du flou, restauration par déconvolution et évaluation quantitative.

Le projet est entièrement réalisé sous forme de **notebook Python** et couvre l'ensemble d'une chaîne de vision par ordinateur moderne.

---

##  Fonctionnalités

- **Modélisation du flou** via une Point Spread Function (PSF) paramétrée (longueur + angle)
- **Génération de dataset synthétique** : plaques réalistes avec flou de mouvement contrôlé
- **Analyse fréquentielle** (FFT 2D) pour estimer les paramètres du flou
- **Estimation CNN du PSF** : réseau léger entraîné sur le log-spectre FFT
- **3 méthodes de déconvolution** : Inverse naïve, Wiener, Tikhonov
- **Benchmark automatisé** avec métriques PSNR et SSIM
- **Pipeline final automatisé** de bout en bout

---

##  Technologies

| Outil | Usage |
|---|---|
| Python | Langage principal |
| NumPy | Calcul matriciel et convolution |
| OpenCV | Traitement d'images |
| scikit-image | Métriques PSNR / SSIM |
| Matplotlib | Visualisation des résultats |
| PyTorch | Réseau CNN d'estimation du PSF |

---

##  Architecture du projet

```
 notebook principal (10 cellules)
│
├── 1. PSFGenerator          — Génération des noyaux de flou
├── 2. Dataset synthétique   — Plaques nettes + floues annotées
├── 3. FrequencyAnalyzer     — Analyse FFT, estimation du flou
├── 4. PSFEstimatorCNN       — Réseau convolutif de régression [L, θ]
├── 5. DeconvolutionEngine   — Inverse, Wiener, Tikhonov
├── 6. QualityEvaluator      — PSNR & SSIM
├── 7. BenchmarkEvaluator    — Évaluation sur 80 images
└── 8. run_full_pipeline()   — Pipeline automatisé end-to-end
```

---

##  Modèle de dégradation

Le flou est modélisé par :

```
image_floue = image_nette ∗ PSF + bruit
```

où la PSF est un noyau de convolution normalisé défini par :
- **L** — longueur du déplacement
- **θ** — angle de la direction du mouvement

---

##  Résultats

Les méthodes ont été comparées sur **80 images** via PSNR et SSIM :

| Méthode | Qualité | Stabilité |
|---|---|---|
| Inverse naïve |  Variable, souvent bruitée |  Faible |
| **Wiener** |  **Meilleures performances** |  Bonne |
| Tikhonov |  Légèrement plus lisse |  Très stable |

> **Conclusion** : le filtre de Wiener offre le meilleur compromis qualité/stabilité.

---

##  Lancement

### Prérequis

```bash
pip install numpy opencv-python scikit-image matplotlib torch torchvision
```

### Exécution

Ouvrir et exécuter le notebook cellule par cellule, ou lancer le pipeline complet :

```python
run_full_pipeline(image_path="path/to/blurry_plate.png")
```

---
