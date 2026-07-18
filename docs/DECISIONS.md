# Architecture Decisions

## ADR-001 — CPU-first statistical baseline

**Decision:** เริ่มด้วย streaming per-pixel normal distribution

**Why:** รันได้โดยไม่ดาวน์โหลด model, อธิบายง่าย, เป็น minimum bar ที่ reproducible และเปิดเผยข้อจำกัด
ด้าน alignment/lighting ได้ชัดเจน

**Trade-off:** ความแม่นยำและ invariance ต่ำกว่า embedding/PatchCore จึงใช้เป็น baseline ไม่ใช่ final production model

## ADR-002 — TOML configuration

**Decision:** ใช้ TOML ผ่าน Python standard library

**Why:** ลด dependency และรองรับ typed validation

**Trade-off:** ผู้ใช้ Python 3.10 หรือต่ำกว่าต้อง upgrade; project กำหนด Python 3.11+

## ADR-003 — Safe array artifact

**Decision:** ใช้ NPZ + JSON + SHA-256 manifest และปิด pickle

**Why:** ลด remote-code-execution surface จาก unsafe deserialization และตรวจ corruption ได้

**Trade-off:** ยังไม่ใช่ signed artifact และไม่เหมาะกับ arbitrary model graphs; model ใหม่ต้องมี serializer ของตนเอง

## ADR-004 — Optional heavy integrations

**Decision:** Gradio และ Anomalib ไม่อยู่ใน core dependencies

**Why:** ลด build time, CVE surface, GPU wheel mismatch และทำให้ unit tests รันบน CPU ได้เร็ว

**Trade-off:** ผู้ใช้ต้องติดตั้ง extra ที่ตรงกับ hardware และทดสอบ compatibility เพิ่มก่อนใช้งาน
