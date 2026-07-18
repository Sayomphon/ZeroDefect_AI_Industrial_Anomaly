# Notebook plan

Notebooks ใช้สำหรับ exploration และ storytelling เท่านั้น ส่วน reusable logic ต้องอยู่ใน `src/zerodefect_ai`

Notebook ที่พร้อมใช้งาน:

- `00_colab_quickstart.ipynb` — clean runtime setup, synthetic smoke, evidence display และ
  authenticated opt-in Gradio share

เปิดจาก `main`:

https://colab.research.google.com/github/Sayomphon/ZeroDefect_AI_Industrial_Anomaly/blob/main/notebooks/00_colab_quickstart.ipynb

ลำดับที่วางไว้สำหรับ experiment phase:

1. `01_eda_and_baseline.ipynb`
2. `02_model_experiments.ipynb`
3. `03_evaluation.ipynb`
4. `04_demo.ipynb`

Notebook ต้องไม่มี saved output/secret และต้องเรียก reusable functions จาก package แทนการคัดลอก
model logic เพื่อไม่สร้าง hidden notebook state
