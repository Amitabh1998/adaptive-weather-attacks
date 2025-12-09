# Adaptive Weather Attacks

**Adversarial Weather Attacks on Traffic Sign Recognition using Variable Classifier-Free Guidance**

This project implements weather-based adversarial attacks on traffic sign classifiers using diffusion models with Variable CFG (Classifier-Free Guidance) scheduling.

## Key Contribution

Traditional diffusion-based attacks use constant CFG throughout the denoising process. We show that **variable CFG schedules** (linear decay, cosine decay) produce:
- More realistic weather perturbations (lower LPIPS)
- Better attack transferability across model architectures
- Comparable or better attack success rates

## Project Structure

```
adaptive-weather-attacks/
├── src/                    # Core Python modules
│   ├── data/              # Dataset loading & transforms
│   ├── models/            # Model architectures & training
│   ├── attacks/           # Pixel-based attacks (FGSM, PGD, CW)
│   ├── diffusion/         # V-CFG implementation (main contribution)
│   ├── metrics/           # ASR, LPIPS, SSIM evaluation
│   └── utils/             # Visualization & utilities
├── notebooks/             # Colab notebooks (run experiments here)
├── configs/               # Experiment configurations
├── checkpoints/           # Saved model weights
├── results/               # Experiment outputs
└── outputs/               # Generated adversarial images
```

## Quick Start (Google Colab)

```python
# Cell 1: Clone and setup
!git clone https://github.com/YOUR_USERNAME/adaptive-weather-attacks.git
%cd adaptive-weather-attacks
!pip install -e . -q

# Cell 2: Mount Drive and copy dataset
from google.colab import drive
drive.mount('/content/drive')

import shutil, os
if not os.path.exists('/content/GTSRB_dataset'):
    shutil.copytree('/content/drive/MyDrive/GTSRB_dataset', '/content/GTSRB_dataset')

# Cell 3: Run experiments
from src.data import get_dataloaders
from src.models import get_model
from src.diffusion import VariableCFGAttack

# Load data and models
_, _, test_loader = get_dataloaders('/content/GTSRB_dataset')
model = get_model('resnet50', num_classes=43, pretrained=False)

# Run V-CFG attack
attack = VariableCFGAttack(cfg_schedule='linear')
results = attack.run(test_loader, num_samples=100)
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_train_classifiers.ipynb` | Train ResNet-50, EfficientNet-B0, ViT on GTSRB |
| `02_baseline_attacks.ipynb` | Run FGSM, PGD, CW attacks |
| `03_weather_attacks.ipynb` | Generate weather perturbations with diffusion |
| `04_vcfg_experiments.ipynb` | **Main experiment**: Compare CFG schedules |
| `05_results_analysis.ipynb` | Generate figures and analysis |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (A100 recommended)
- ~10GB GPU memory for Stable Diffusion

## Dataset

This project uses the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_news.html):
- 43 classes of traffic signs
- ~39,000 training images
- ~12,600 test images

## Citation

If you use this code, please cite:

```bibtex
@misc{adaptive-weather-attacks-2024,
  author = {Your Name},
  title = {Adaptive Weather Attacks: Variable CFG Diffusion for Adversarial Traffic Sign Recognition},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/adaptive-weather-attacks}
}
```

## License

MIT License
