# Roadmap

## Phase 1 — Runnable baseline MVP

- secure image/data pipeline
- statistical normal-only detector
- artifact save/load + integrity check
- threshold calibration และ business cost
- image-level/pixel-level evaluation foundation
- CLI, synthetic smoke workflow และ optional Gradio UI
- automated tests และ operator documentation

## Phase 2 — PatchCore benchmark

- เพิ่ม `PatchCoreDetector` adapter ผ่าน Anomalib
- lock PyTorch backend ตาม CPU/CUDA target
- MVTec category experiment: statistical baseline vs PatchCore
- coreset 1%/5%/10%, resolution และ backbone ablation
- pixel AUROC/AUPR/PRO พร้อม defect-type slices
- latency, peak RAM/VRAM และ artifact-size benchmark

## Phase 3 — Factory pilot

- camera/lighting/fixture SOP และ golden image validation
- edge inference container และ PLC/MES/QMS integration contract
- site/line/SKU-specific threshold policy
- borderline human-review queue และ feedback capture
- brightness/focus/position/score-distribution drift monitors
- signed model registry, authentication, audit/retention และ rollback drill

## Exit criteria ก่อนอ้าง production readiness

- validation บนภาพจากกล้อง กระบวนการ และ defect taxonomy ของโรงงานเป้าหมาย
- agreed false-reject/escape cost และ operating point กับ Quality/Production
- robustness test ครอบคลุม shift ที่คาดการณ์ได้
- security/privacy/governance review ผ่าน
- monitoring, fail-safe, manual override, rollback และ incident runbook ถูกทดสอบ
