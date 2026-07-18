# Google Colab Readiness

วันที่ตรวจ: 2026-07-18

## Supported workflow

`notebooks/00_colab_quickstart.ipynb` เป็น clean-runtime entry point สำหรับ Phase 1 CPU-first MVP:

1. clone หรือ fast-forward public repository ที่ branch `main`
2. ติดตั้ง package พร้อม optional Gradio dependency ผ่าน `pip install .[demo]`
3. ตรวจ Python, NumPy, Pillow และ GPU command availability
4. รัน `run_smoke()` ซึ่งสร้าง synthetic MVTec-like data, train, calibrate, save/load artifact,
   evaluate และสร้าง overlay
5. แสดง metrics และ prediction evidence
6. เปิด authenticated Gradio share แบบ explicit opt-in เท่านั้น

Notebook ไม่คัดลอก detector หรือ inference logic แต่เรียก `ProjectConfig`, `run_smoke()` และ
`launch_colab_app()` จาก codebase เพื่อให้ CLI, test และ notebook ใช้ behavior เดียวกัน

## Gradio exposure model

Managed Colab runtime ไม่สามารถให้ browser เข้าถึง localhost โดยตรง Gradio จึงต้องสร้าง public
share tunnel สำหรับ UI ใน notebook ระบบเลือกแนวทางต่อไปนี้:

- `ENABLE_PUBLIC_GRADIO = False` เป็น default เพื่อให้ Run all ไม่เปิด internet-facing endpoint
- ผู้ใช้ต้องเปลี่ยน flag, กรอก username/password ชั่วคราว และยืนยันผ่าน
  `confirm_public_share=True`
- password ต้องมีอย่างน้อย 12 ตัวอักษร
- จำกัด `max_threads=4`, ไม่แสดง internal error และไม่ expose filesystem path เพิ่ม
- ห้ามใช้ confidential factory image, PII, secret หรือ proprietary artifact

Local `app/app.py` ยังรักษา default `127.0.0.1` และ `share=False` เดิม

## Validation gates

- notebook เป็น valid nbformat 4 JSON
- code cells ไม่มี saved output หรือ execution count
- ทุก code cell compile ได้
- clean-sequence executor รันทุก cell โดย Gradio share ปิดอยู่
- core automated tests และ Colab launch-policy tests ผ่าน
- optional Gradio environment สร้าง `Blocks` จาก valid synthetic artifact ได้
- local loopback Gradio HTTP smoke ผ่านโดยไม่สร้าง public share

ผลคำสั่งและ environment จริงบันทึกใน `docs/TEST_REPORT.md` และ `docs/BUILD_LOG.md`

## Known limitations

- synthetic smoke ยืนยัน plumbing เท่านั้น ไม่ใช่ผล benchmark บน MVTec AD หรือกล้องโรงงาน
- hosted Colab resource, package image และ GPU availability เปลี่ยนได้ จึงต้อง rerun clean runtime
  ก่อน demo สำคัญ
- Gradio share เป็น demo path ไม่ใช่ production deployment
- Phase 2 PatchCore/Anomalib benchmark และ real-dataset experiment ยังอยู่ใน roadmap
