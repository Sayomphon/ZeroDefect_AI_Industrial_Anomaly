# Test and Verification Report

วันที่ตรวจ: 2026-07-17
Runtime หลัก: Python 3.12.13, NumPy 2.3.5, Pillow 12.2.0

## Summary

- Automated tests: **18 passed, 0 failed**
- Python compilation: **passed**
- End-to-end CLI smoke: **passed**
- Wheel packaging/install/import: **passed**
- Basic credential and unsafe-pattern scan: **no matches**
- Maximum source line length check: **no lines over 100 characters**

## Commands executed

```bash
PYTHONPATH=src python -m compileall -q src tests app
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m zerodefect_ai smoke \
  --config configs/base.toml \
  --output-dir /private/tmp/zerodefect_smoke_20260717
python -m pip install . --no-deps --no-build-isolation \
  --target /private/tmp/zerodefect_install_20260717_v1
```

## Automated coverage by behavior

- config file parsing, unknown-key rejection และ symlink rejection
- detector pre-fit guard, shape validation และ anomaly-score separation
- normal quantile, F1 และ high escape-cost calibration
- AUROC, average precision และ threshold decision metrics
- artifact save/load round trip และ checksum tamper detection
- image resize, symlink rejection, non-finite pixels, single-channel conversion และ integer range
- synthetic train → save/load → MVTec evaluation → overlay prediction

## Synthetic smoke evidence

Synthetic set ประกอบด้วย normal training 12, normal validation 12, normal test 4 และ defect test 4
ผลหลังแยก calibration set:

- image ROC AUC: 1.0
- image average precision: 1.0
- image F1: 1.0
- false rejects: 0/4
- defect escapes: 0/4
- pixel ROC AUC/AUPR: 1.0/1.0 บน 65,536 sampled pixels
- detector-only latency: mean approximately 0.26 ms/image บน runtime ที่ใช้ทดสอบ

ตัวเลขนี้ยืนยัน deterministic plumbing กับ defect สังเคราะห์ที่แยกง่ายเท่านั้น ห้ามใช้เป็นหลักฐาน
ประสิทธิภาพกับ MVTec AD, camera images หรือ defect จริงของโรงงาน

## Issue found and corrected

Smoke รอบแรก calibrate ด้วย training images ชุดเดียวกัน ทำให้ normal test score สูงกว่า in-sample range
และเกิด False Reject 100% แม้ AUROC เท่ากับ 1.0 การแก้ไขคือสร้าง normal validation แยกและเพิ่ม
`--calibration-normal-dir` ใน documented training workflow

## Graphify audit

- Final graph: 340 nodes, 719 graph edges, 17 communities
- Initial raw extraction warning: 138 dangling-endpoint edges และ 43 same-endpoint collapsed edges
- Final post-merge diagnostic: 0 dangling, missing, self-loop หรือ same-endpoint collapsed edges
- Incremental diff: 34 new nodes, 53 new edges, 13 removed nodes และ 34 removed edges

Graph ใช้เป็น navigation aid ไม่ใช่หลักฐาน formal completeness ของ dependency/call graph เพราะ edge
ที่ extraction producer suppress ก่อน merge อาจไม่ปรากฏใน post-merge diagnostic

## Checks not executed in this environment

- Ruff, mypy, Bandit และ coverage percentage เพราะ dev tools ยังไม่ได้ติดตั้ง
- real MVTec evaluation เพราะไม่มี dataset ใน workspace
- PatchCore/Anomalib benchmark เพราะไม่มี model stack และ hardware-specific PyTorch backend
- Gradio browser test เพราะ Gradio เป็น optional dependency และยังไม่ได้ติดตั้ง
- container, SBOM, vulnerability scan, load test และ penetration test

ก่อน merge/release ให้ติดตั้ง `pip install -e ".[dev]"` แล้วรัน quality commands ใน README ผ่าน CI

## Colab and Gradio verification — 2026-07-18

Clean virtual environment:

- Python 3.12.13
- NumPy 2.5.1
- Pillow 12.3.0
- Gradio 6.20.0

ผลตรวจ:

- clean installation `pip install ".[demo]"`: passed
- automated tests: **22 passed, 0 failed, 0 skipped** เมื่อมี Gradio dependency
- notebook JSON/compile/no-output validation: passed
- notebook sequential Run-all simulation: passed โดย Gradio public share ปิดตาม default
- Gradio real `Blocks` build จาก synthetic artifact: passed
- Gradio loopback launch ด้วย `share=False`: HTTP 200 และปิด server หลังตรวจ
- Colab launch-policy tests: explicit consent, authentication, minimum password และ thread limit ผ่าน
- Ruff สำหรับไฟล์ Colab/Gradio ที่เพิ่มหรือแก้: passed
- mypy สำหรับ `app` และ `scripts` โดย skip third-party import bodies: passed
- Bandit สำหรับ `app` และ `scripts`: no findings
- pytest branch coverage ของ core package: 75%

ระหว่าง notebook execution รอบแรกพบและแก้ compatibility issues 3 รายการ:

1. `find_spec("google.colab")` อาจโยน `ModuleNotFoundError` เมื่อไม่มี namespace `google`
2. local validator อาจไม่มี IPython จึงเพิ่ม text fallback สำหรับ evidence display
3. script execution ไม่ได้เพิ่ม repository root ใน `sys.path` จึง import `app.app` ไม่ได้

ไม่ได้สร้าง public Gradio URL ระหว่าง automated test เพราะเป็น internet-facing endpoint การทดสอบ
Colab-specific path จึงแยกเป็น real local UI build/HTTP test และ launch-policy test ที่ตรวจว่า hosted
Colab path ใช้ `share=True`, inline display, debug output, authentication และ resource limit ตามที่กำหนด

Full-project Ruff และ mypy ยังมี modernization/type findings เดิมใน Phase 1 core ซึ่งไม่เกิดจากงาน Colab
รอบนี้ ไฟล์ที่เพิ่มหรือแก้ในรอบนี้ผ่าน scoped quality gates ข้างต้น
