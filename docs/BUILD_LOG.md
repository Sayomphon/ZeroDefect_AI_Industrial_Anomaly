# Build Log

เอกสารนี้บันทึกสิ่งที่ดำเนินการกับโปรเจค เหตุผล และหลักฐานการตรวจสอบ เพื่อให้ทีม Business,
Developer, Data Engineer และ AI Engineer สามารถติดตามสถานะเดียวกันได้

## 2026-07-17 — Project bootstrap

### Inputs reviewed

- อ่าน `03_ZeroDefect_AI_Industrial_Anomaly_Plan.docx` ครบทั้ง 15 หน้า
- ยืนยันว่า workspace เริ่มต้นมีเพียง planning document และยังไม่มี source code หรือ Git repository

### Decisions made

- กำหนด Phase 1 เป็น runnable CPU-first MVP แทนการติดตั้ง PatchCore/GPU stack ทันที
- ใช้ Python 3.11+ และลด core dependencies เหลือ NumPy กับ Pillow
- ใช้ TOML config เพื่อไม่เพิ่ม YAML parser ใน production core
- แยก model score, calibration policy และ QC decision เป็นคนละ layer
- ป้องกัน unsafe artifact deserialization ด้วย NPZ ที่ปิด pickle และ JSON schema validation
- กำหนด Gradio และ Anomalib เป็น optional dependencies

### Files scaffolded

- Packaging/config: `pyproject.toml`, `requirements*.txt`, `configs/base.toml`
- Operator entrypoint: `README.md`
- Governance: `.gitignore`, `docs/ARCHITECTURE.md`, `docs/SECURITY.md`,
  `docs/DATA_CONTRACT.md`, `docs/DECISIONS.md`, `docs/ROADMAP.md`

### Verification status

- Completed; detailed evidence is recorded in `docs/TEST_REPORT.md`

## 2026-07-17 — Phase 1 implementation

### Core implementation

- เพิ่ม strict TOML configuration และ resource limits
- เพิ่ม secure image decoding, deterministic preprocessing และ MVTec-compatible discovery
- เพิ่ม streaming statistical detector, anomaly map และ upper-quantile image score
- เพิ่ม normal-only, F1 และ business-cost threshold calibration functions
- เพิ่ม image-level AUROC/AUPR/F1/confusion/cost metrics และ memory-bounded pixel metrics
- เพิ่ม safe NPZ/JSON artifact persistence, SHA-256 integrity verification และ atomic writes
- เพิ่ม reusable inspection service, heatmap renderer, CLI และ optional Gradio UI
- เพิ่ม synthetic MVTec-like generator และ end-to-end smoke workflow

### Automated tests

- เขียน 18 tests ครอบคลุม configuration, detector, calibration, metrics, artifact tampering,
  secure image input และ end-to-end workflow
- รอบแรกพบ `compare_digest` import ผิด module ทำให้ artifact load ล้มเหลว 3 tests
- แก้จาก `hashlib.compare_digest` เป็น `hmac.compare_digest` แล้วรันใหม่ผ่าน 18/18 tests
- เพิ่ม test สำหรับ single-channel input และ integer pixel ที่เกิน byte range

### Calibration correction discovered by smoke test

- Smoke รอบแรกใช้ training images ซ้ำเพื่อ calibrate threshold และพบ False Reject 100% บน normal test
  แม้ ranking AUROC เป็น 1.0 ซึ่งยืนยันว่า ranking metric ที่ดีไม่ได้แปลว่า operating point ใช้งานได้
- ปรับ synthetic workflow ให้มี normal validation แยกจาก training
- Smoke รอบถัดไปได้ False Reject 0%, Defect Escape 0% และ F1 1.0 บน synthetic test
- ผลดังกล่าวใช้ยืนยัน pipeline plumbing เท่านั้น ไม่ใช่ model-quality claim

### Packaging and security checks

- build Python wheel สำเร็จและ import CLI จาก isolated target directory ได้
- `compileall` ผ่านสำหรับ `src`, `tests` และ `app`
- manual unsafe-pattern scan ไม่พบ hard-coded credential, private key, pickle execution,
  `shell=True`, public Gradio share หรือ `os.system`
- ตรวจ line length ตาม project limit 100 characters แล้วไม่พบ violation
- cleanup เฉพาะ generated build, egg-info และ cache intermediates หลังตรวจ packaging

### Architecture graph audit

- Graphify corpus: 39 files, approximately 9,292 words; sensitive file 1 ไฟล์ถูกข้าม
- Final incremental graph: 340 nodes, 719 graph edges, 17 labeled communities
- Outputs: `graphify-out/graph.json`, `graphify-out/graph.html`,
  `graphify-out/GRAPH_REPORT.md`
- Initial raw extraction warning: 138 dangling-endpoint edges และ 43 same-endpoint collapsed edges
  จาก AST/type/call extraction; final post-merge graph ไม่มี dangling/collapsed edge ที่เหลืออยู่ แต่ข้อมูล
  ที่ producer suppress ไปก่อน build ไม่สามารถกู้คืนได้ จึงยังไม่ควรถือว่า graph สมบูรณ์
- Incremental refresh หลัง final documentation: 34 nodes/53 edges เพิ่ม, 13 nodes/34 edges ถูกแทนที่
- Graphify recorded 0 external LLM input/output tokens; semantic extraction ใช้ session agent

### Scope intentionally deferred

- ไม่ดาวน์โหลด MVTec AD หรือ pretrained model อัตโนมัติ
- ยังไม่ติดตั้ง/benchmark PatchCore, Anomalib, PyTorch หรือ GPU backend
- Ruff, mypy และ Bandit ถูกกำหนดใน dev dependencies แต่ runtime ปัจจุบันไม่มีเครื่องมือเหล่านี้
  จึงยังไม่ได้รัน; ต้องรันใน CI/virtual environment หลังติดตั้ง `.[dev]`
- ยังไม่มี production authentication, signed artifact registry, container hardening หรือ OT integration

## 2026-07-17 — GitHub publication preparation

### Remote inspection

- Target: `Sayomphon/ZeroDefect_AI_Industrial_Anomaly`
- Remote default branch: `main`; repository owner permission: `ADMIN`
- Remote เดิมมี placeholder `README.md` และ Apache License 2.0 เท่านั้น

### Publication scope

- รักษา Apache `LICENSE` จาก remote และกำหนด `pyproject.toml` ให้ใช้งาน license เดียวกัน
- แทนที่ placeholder README ด้วย project README ที่ผ่านการตรวจแล้ว
- รวม source, config, app, tests และ Markdown documentation
- ไม่รวม planning DOCX, generated artifacts, local data, outputs, secrets หรือ `graphify-out`
- Publish ผ่าน branch `agent/initialize-zerodefect-ai`; ไม่เขียนทับ `main` และไม่ force-push

### Publication result

- สร้าง implementation commit `58b75d1` ด้วยข้อความ `Initialize ZeroDefect AI MVP`
- Push สำเร็จไปยัง `origin/agent/initialize-zerodefect-ai` และตั้งค่า upstream tracking แล้ว
- ตรวจยืนยันว่า local `HEAD` และ remote branch ชี้ไปยัง commit เดียวกันหลัง push
- ไม่สร้าง Pull Request เนื่องจากขอบเขตงานรอบนี้กำหนดให้ commit และ push เท่านั้น
- เพิ่มบันทึกผลการเผยแพร่นี้เป็น documentation follow-up commit บน branch เดิม

## 2026-07-18 — Colab delivery and main-branch integration

### Colab notebook

- เพิ่ม `notebooks/00_colab_quickstart.ipynb` เป็น clean-runtime entry point
- notebook clone/pull `main` แบบไม่เขียนทับ dirty checkout และติดตั้ง `.[demo]`
- เรียก reusable `ProjectConfig` และ `run_smoke()` แทนการคัดลอก model logic
- แสดง runtime diagnostics, smoke report, image metrics และ prediction overlay
- ไม่มี saved cell output, credential หรือ hard-coded local path

### Colab-specific Gradio control

- เพิ่ม `launch_colab_app()` โดยต้อง opt in ต่อ public share อย่างชัดเจน
- บังคับ username และ password อย่างน้อย 12 ตัวอักษร
- จำกัด worker threads และคง local launch default เป็น loopback/ไม่ share
- notebook ปิด Gradio share โดย default เพื่อให้ Run all ไม่เปิด endpoint

### Validation and GitHub delivery

- เพิ่ม standard-library notebook validator และ Colab launch-policy tests
- บันทึกผล validation จริงใน `docs/TEST_REPORT.md`
- publish ผ่าน branch `agent/initialize-zerodefect-ai` แล้ว merge เข้า `main` ผ่าน GitHub PR
