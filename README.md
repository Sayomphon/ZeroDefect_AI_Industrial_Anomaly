# ZeroDefect AI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sayomphon/ZeroDefect_AI_Industrial_Anomaly/blob/main/notebooks/00_colab_quickstart.ipynb)

Cold-start industrial visual anomaly detection for factories with abundant normal-product images and limited defect data. This MVP uses an explainable normal-only statistical baseline to produce anomaly heatmaps and explicitly separates the **model score** from the **business decision threshold**.

> Status: Phase 1 runnable MVP covering the data contract, baseline detector, calibration, evaluation, and demo UI.
> It is not yet production line-ready, and the core installation does not include the PatchCore runtime.

## Capabilities

- Fits a normal-only baseline with streaming mean and variance, without loading the full dataset into RAM.
- Inspects images at image level and produces pixel-level anomaly maps.
- Calibrates thresholds using a normal-score quantile, labelled F1, or expected business cost.
- Stores artifacts as NPZ/JSON with a SHA-256 integrity manifest, without pickle.
- Reads the MVTec AD layout for evaluation by defect type.
- Runs an end-to-end synthetic-data smoke test without a GPU or model download.
- Provides an optional Gradio inspection UI.

## Quick start

Python 3.11 or later is required. Creating a virtual environment is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the smoke test:

```bash
zerodefect smoke --output-dir outputs/smoke
```

Alternatively, run it without installing the package:

```bash
PYTHONPATH=src python -m zerodefect_ai smoke --output-dir outputs/smoke
```

## Google Colab quickstart

Open [`notebooks/00_colab_quickstart.ipynb`](notebooks/00_colab_quickstart.ipynb) with the **Open in Colab** button, then select **Runtime → Run all**. The notebook clones the `main` branch, installs `.[demo]`, runs the synthetic smoke workflow, and displays a prediction overlay without requiring a GPU.

Public Gradio sharing is disabled by default. To expose the UI, explicitly opt in from the final cell and provide temporary credentials. Never use real factory images or confidential data with a public share URL. See [`docs/COLAB_READINESS.md`](docs/COLAB_READINESS.md) for validation details and limitations.

Train from a folder of normal images:

```bash
zerodefect train \
  --normal-dir /path/to/category/train/good \
  --calibration-normal-dir /path/to/category/validation/good \
  --artifact-dir artifacts/statistical-v1 \
  --config configs/base.toml
```

Use a normal validation set separate from training for threshold calibration. If `--calibration-normal-dir` is omitted, the training set is reused and the artifact records `training_set_reuse`. This is suitable only for smoke tests and early exploration because it can understate the false-reject rate.

Inspect a single image:

```bash
zerodefect predict \
  --artifact-dir artifacts/statistical-v1 \
  --image /path/to/image.png \
  --output-dir outputs/prediction
```

Evaluate an MVTec AD category:

```bash
zerodefect evaluate-mvtec \
  --artifact-dir artifacts/statistical-v1 \
  --dataset-root /path/to/mvtec \
  --category bottle \
  --output-json outputs/bottle-metrics.json
```

Launch the demo UI:

```bash
python -m pip install -e ".[demo]"
ZERO_DEFECT_ARTIFACT_DIR=artifacts/statistical-v1 python app/app.py
```

The UI binds to `127.0.0.1` and disables public sharing by default. If it must be exposed over a network, place it behind an authenticated reverse proxy and enforce appropriate resource limits.

## Dataset contract

Core training accepts a directory of normal images directly. MVTec evaluation expects this layout:

```text
<dataset-root>/<category>/
├── train/good/*
├── test/good/*
├── test/<defect-type>/*
└── ground_truth/<defect-type>/*_mask.png
```

The project does not download or commit datasets automatically. Before publishing or using the system commercially, review the license and terms for every dataset and model.

## Architecture

```text
Image files
   │  validation: path, bytes, format, decoded pixels
   ▼
Secure Image I/O ──► deterministic preprocessing
   ▼
Statistical Detector ──► anomaly map ──► image score
   │                                         │
   └──────── versioned NPZ/JSON artifact ◄───┘
                                             ▼
                              calibrated business threshold
                                             ▼
                                  QC decision + heatmap
```

For details, see [the architecture](docs/ARCHITECTURE.md), [security and governance](docs/SECURITY.md), and [the data contract](docs/DATA_CONTRACT.md).

## Quality checks

```bash
PYTHONPATH=src python -m compileall -q src tests app
PYTHONPATH=src python -m unittest discover -s tests -v
ruff check .
mypy src
bandit -q -r src app
```

## PatchCore extension path

The next phase will add a detector adapter based on `anomalib.models.Patchcore` after the hardware-specific PyTorch backend and the benchmark dependency set have been fixed. It is intentionally excluded from the core installation to reduce the installation surface, build time, and risk of GPU/CPU wheel mismatches. The detector, artifact, and decision layers are already separated, so a new model can be introduced without changing the main CLI contract.

The intended implementation path follows the [Anomalib PatchCore documentation](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/patchcore.html).

## Security notes

- Artifacts are never loaded with pickle (`numpy.load(..., allow_pickle=False)`).
- File bytes, decoded pixels, formats, and image dimensions are bounded.
- Dataset discovery does not follow symlinks.
- Secrets are not hard-coded, and Gradio public sharing is disabled by default.
- The Colab Gradio helper requires explicit consent for public exposure and authentication before it creates a share URL.
- The SHA-256 manifest detects accidental corruption but is not a digital signature; production deployments should add signing.
- Do not commit real factory images, PII, secrets, or proprietary models to a public repository.

## Project records

- [`docs/BUILD_LOG.md`](docs/BUILD_LOG.md) — work completed and verification evidence.
- [`docs/TEST_REPORT.md`](docs/TEST_REPORT.md) — automated-test and smoke-test results, plus evidence limitations.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — concise Architecture Decision Records.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — work remaining after the MVP.

## License

Released under the [Apache License 2.0](LICENSE) specified by this repository. Dataset, pretrained-model, and third-party-dependency licenses and terms must still be reviewed separately.
