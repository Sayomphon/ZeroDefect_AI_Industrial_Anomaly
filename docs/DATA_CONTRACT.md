# Data Contract

## Training input

- ข้อมูลต้องเป็นภาพ normal เท่านั้นสำหรับ statistical detector รอบนี้
- format ที่ยอมรับตาม config: PNG, JPEG และ BMP
- ทุกภาพถูกแปลงเป็น RGB และขนาดเดียวกันด้วย preprocessing config ที่บันทึกใน artifact
- dataset discovery ไม่ follow symlink และเรียง path เพื่อ reproducibility
- แนะนำให้เก็บ manifest ภายนอกที่มี source, capture time, camera, lot, product/SKU และ checksum

## MVTec evaluation input

```text
<root>/<category>/train/good/<image>
<root>/<category>/test/good/<image>
<root>/<category>/test/<defect_type>/<image>
<root>/<category>/ground_truth/<defect_type>/<stem>_mask.png
```

`good` map เป็น label `0`; defect type อื่น map เป็น label `1` การประเมินจะรายงาน overall metrics
และ slice count/score แยก defect type เพื่อไม่ซ่อน performance ที่แย่ในบาง defect

## Model input

- type: NumPy `float32`
- shape: `[height, width, 3]`
- value range: `[0.0, 1.0]`
- color: RGB
- NaN/Inf: ไม่อนุญาต

## Prediction output

- `score`: finite non-negative float
- `threshold`: finite float จาก versioned artifact
- `is_anomaly`: `score >= threshold`
- `anomaly_map`: `float32 [height, width]`
- `model_type`, `artifact_schema_version` และ preprocessing config สำหรับ traceability

## Leakage control

ห้ามใช้ test defect label เพื่อ tune detector หากต้องการอ้าง pure unsupervised setting การใช้ defect labels
เพื่อเลือก F1/cost threshold ต้องระบุเป็น semi-supervised calibration และแยก validation/test อย่างชัดเจน
