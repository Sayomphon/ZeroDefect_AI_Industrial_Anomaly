# System Architecture

## Context

ZeroDefect AI แก้ cold-start visual inspection ซึ่งมี normal images มากกว่า defect examples
ดังนั้นระบบต้องเริ่มจาก normal-only learning ได้ ขณะเดียวกันต้องไม่ผูก model metric เข้ากับ
business decision แบบตายตัว เพราะแต่ละโรงงานมีต้นทุน False Reject และ Defect Escape ต่างกัน

## Component flow

1. **Dataset discovery** ค้นหาเฉพาะไฟล์ภาพที่อนุญาต ไม่ follow symlink และเรียงลำดับแบบ deterministic
2. **Secure image I/O** ตรวจ file size, extension, decoded format และ pixel count ก่อนแปลงเป็น RGB tensor
3. **Preprocessing** center-crop หรือ stretch ตาม versioned config แล้ว normalize เป็น `float32 [0, 1]`
4. **Detector** เรียนรู้ pixel-wise normal distribution แบบ streaming และคืน anomaly map เป็น z-deviation
5. **Scoring** สรุป image score ด้วย upper quantile เพื่อลด sensitivity ต่อ single noisy pixel
6. **Calibration** เลือก decision threshold จาก normal-only quantile, F1 หรือ expected business cost
7. **Artifact store** บันทึก model arrays, metadata, config และ threshold พร้อม SHA-256 manifest
8. **Service/UI** โหลด immutable artifact แล้วคืน score, decision และ heatmap โดยไม่ train ใน request path
9. **Evaluation** วัด image-level AUROC/AUPR/F1, confusion matrix, cost และ slice ตาม defect type

## Dependency direction

```text
CLI / Gradio App
       │
       ▼
Inspection Service / Workflows
       │
       ├── Dataset + Secure Image I/O
       ├── Detector interface + Statistical detector
       ├── Calibration + Metrics
       ├── Visualization
       └── Artifact repository
```

Core modules ไม่ import Gradio, Anomalib หรือ framework ฝั่ง UI ทำให้สามารถนำ service layer
ไปใช้กับ FastAPI, batch job, edge process หรือ OT integration ได้โดยไม่เปลี่ยน model code

## Production evolution

- **POC:** local CLI/Gradio, filesystem artifact, single process
- **MVP:** containerized inference service, signed artifact, CI, batch evaluation, authentication
- **Production:** camera/PLC integration, model registry, canary/rollback, drift monitoring, audit trail,
  human review queue, site-specific calibration และ SLA/SLO

## Known design limits

- Statistical baseline ต้องการตำแหน่งชิ้นงานและ illumination ที่ค่อนข้างคงที่
- Heatmap บอกความต่างจาก normal distribution ไม่ใช่ causal explanation
- Pixel threshold แยกจาก image decision threshold; MVP ยังไม่ calibrate segmentation mask สำหรับ production
- Benchmark dataset ไม่แทน camera/lighting/process validation บนสายการผลิตจริง
