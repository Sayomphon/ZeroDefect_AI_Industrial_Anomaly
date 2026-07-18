# ZeroDefect AI

Cold-start industrial visual anomaly detection สำหรับโรงงานที่มีภาพสินค้าปกติจำนวนมาก
แต่ข้อมูล defect มีจำกัด ระบบรอบ MVP ใช้ normal-only statistical baseline ที่อธิบายได้
สร้าง anomaly heatmap และแยก **model score** ออกจาก **business decision threshold** อย่างชัดเจน

> สถานะ: Phase 1 runnable MVP สำหรับ data contract, baseline, calibration, evaluation และ demo
> ยังไม่ใช่ production line-ready system และยังไม่ได้รวม PatchCore runtime ไว้ใน core installation

## สิ่งที่ระบบทำได้

- ฝึก baseline จากภาพ normal เท่านั้นด้วย streaming mean/variance โดยไม่โหลด dataset ทั้งหมดเข้า RAM
- ตรวจภาพแบบ image-level และสร้าง pixel-level anomaly map
- calibrate threshold จาก normal quantile, labeled F1 หรือ expected business cost
- เก็บ artifact เป็น NPZ/JSON พร้อม SHA-256 integrity manifest และไม่ใช้ pickle
- อ่าน MVTec AD layout เพื่อ evaluate แยก defect type
- รัน end-to-end smoke test จาก synthetic data ได้โดยไม่ต้องใช้ GPU หรือดาวน์โหลด model
- เปิด Gradio inspection UI ได้เมื่อ install optional dependency

## Quick start

ต้องใช้ Python 3.11 ขึ้นไป แนะนำให้สร้าง virtual environment ก่อนติดตั้ง

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

รัน smoke test:

```bash
zerodefect smoke --output-dir outputs/smoke
```

หรือรันโดยไม่ติดตั้ง package:

```bash
PYTHONPATH=src python -m zerodefect_ai smoke --output-dir outputs/smoke
```

ฝึกจากโฟลเดอร์ภาพ normal:

```bash
zerodefect train \
  --normal-dir /path/to/category/train/good \
  --calibration-normal-dir /path/to/category/validation/good \
  --artifact-dir artifacts/statistical-v1 \
  --config configs/base.toml
```

ควรใช้ normal validation แยกจาก training สำหรับ threshold calibration หากไม่ส่ง
`--calibration-normal-dir` ระบบจะใช้ training set ซ้ำและบันทึก `training_set_reuse` ใน artifact
ซึ่งเหมาะกับ smoke/early exploration เท่านั้น เพราะมีแนวโน้มประเมิน False Reject ต่ำเกินจริง

ตรวจภาพหนึ่งไฟล์:

```bash
zerodefect predict \
  --artifact-dir artifacts/statistical-v1 \
  --image /path/to/image.png \
  --output-dir outputs/prediction
```

ประเมินกับ MVTec AD category:

```bash
zerodefect evaluate-mvtec \
  --artifact-dir artifacts/statistical-v1 \
  --dataset-root /path/to/mvtec \
  --category bottle \
  --output-json outputs/bottle-metrics.json
```

เปิด demo UI:

```bash
python -m pip install -e ".[demo]"
ZERO_DEFECT_ARTIFACT_DIR=artifacts/statistical-v1 python app/app.py
```

UI bind กับ `127.0.0.1` และปิด public sharing โดย default หากต้อง expose ผ่าน network
ให้ใช้ authenticated reverse proxy และกำหนด resource limits เพิ่มเติม

## Dataset contract

Core training รับโฟลเดอร์ที่มีภาพ normal โดยตรง ส่วน MVTec evaluation คาดหวังโครงสร้าง:

```text
<dataset-root>/<category>/
├── train/good/*
├── test/good/*
├── test/<defect-type>/*
└── ground_truth/<defect-type>/*_mask.png
```

โปรเจคไม่ดาวน์โหลดหรือ commit dataset อัตโนมัติ ผู้ใช้งานต้องตรวจ license/terms ของ dataset
และ model ก่อนเผยแพร่หรือใช้เชิงพาณิชย์

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

รายละเอียดอยู่ที่ [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
[docs/SECURITY.md](docs/SECURITY.md) และ [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md)

## Quality checks

```bash
PYTHONPATH=src python -m compileall -q src tests app
PYTHONPATH=src python -m unittest discover -s tests -v
ruff check .
mypy src
bandit -q -r src app
```

## PatchCore extension path

Phase ถัดไปจะเพิ่ม detector adapter ที่ใช้ `anomalib.models.Patchcore` หลังจากล็อก
hardware-specific PyTorch backend และ benchmark dependency set แล้ว เหตุผลที่ไม่ใส่ใน core คือ
ลด installation surface, build time และความเสี่ยงจาก GPU/CPU wheel mismatch โดย interface ของระบบ
แยก detector, artifact และ decision layer ไว้แล้วจึงเพิ่ม model ใหม่ได้โดยไม่เปลี่ยน CLI contract หลัก

อ้างอิง implementation path จาก
[Anomalib PatchCore documentation](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/patchcore.html)

## Security notes

- ไม่โหลด artifact ด้วย pickle (`numpy.load(..., allow_pickle=False)`)
- จำกัด file bytes, decoded pixels, format และ image dimensions
- ไม่ follow symlink ระหว่าง dataset discovery
- ไม่ hard-code secret และไม่เปิด Gradio public share
- SHA-256 manifest ตรวจ accidental corruption ได้ แต่ไม่ใช่ digital signature; production ควรเพิ่ม signing
- ห้ามนำภาพโรงงานจริง, PII, secret หรือ proprietary model ไป commit ใน public repository

## Project records

- [docs/BUILD_LOG.md](docs/BUILD_LOG.md) — สิ่งที่ดำเนินการและหลักฐานการตรวจสอบ
- [docs/TEST_REPORT.md](docs/TEST_REPORT.md) — ผล automated tests, smoke และข้อจำกัดของหลักฐาน
- [docs/DECISIONS.md](docs/DECISIONS.md) — Architecture Decision Records แบบย่อ
- [docs/ROADMAP.md](docs/ROADMAP.md) — งานที่ต้องทำต่อจาก MVP

## License

เผยแพร่ภายใต้ [Apache License 2.0](LICENSE) ตาม license ที่กำหนดไว้ใน repository
การใช้งาน dataset, pretrained model และ third-party dependencies ยังต้องตรวจ license/terms
ของแต่ละรายการแยกต่างหาก
