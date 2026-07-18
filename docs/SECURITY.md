# Security and Governance

## Threat model ของ MVP

ระบบอาจรับ image และ model artifact จาก filesystem หรือ upload UI ความเสี่ยงหลักคือ
resource exhaustion, malformed image, path traversal/symlink, unsafe deserialization,
artifact corruption, accidental data exposure และ unauthenticated network exposure

## Implemented controls

- allowlist image extension และ decoded format
- จำกัด compressed bytes และ decoded pixel count
- เปลี่ยน Pillow decompression-bomb warning เป็น exception
- ปฏิเสธ symlink ใน dataset discovery
- validate shape, dtype และ finite values ก่อนส่งเข้า detector
- โหลด NPZ ด้วย `allow_pickle=False` และจำกัด header/model size
- ตรวจ SHA-256 ของ model/metadata ก่อน load
- เขียน artifact/JSON แบบ atomic เพื่อลด partial-write corruption
- Local Gradio bind เฉพาะ loopback และ `share=False` โดย default
- Colab Gradio helper ต้อง opt in ต่อ public share, ใช้ username/password อย่างน้อย 12 ตัวอักษร
  และจำกัด worker threads
- `.gitignore` ป้องกัน dataset, artifact, `.env` และ output หลุดเข้า repository

## Residual risks

- SHA-256 manifest ที่อยู่ข้าง artifact ป้องกัน accidental corruption แต่ผู้โจมตีที่แก้ทั้ง artifact
  และ manifest ได้ยัง bypass ได้ Production ต้องใช้ artifact signing และ trusted registry
- Process เดียวไม่ใช่ sandbox หากเปิดรับไฟล์จาก untrusted network ควรรัน container แบบ non-root,
  read-only filesystem, CPU/memory/time limits และ malware scanning ตามนโยบายองค์กร
- Local UI ไม่มี authentication ในตัว ต้องวางหลัง authenticated reverse proxy หรือ enterprise gateway
- Colab share URL แม้มี basic authentication ยังเป็น internet-facing endpoint ห้ามใช้กับภาพโรงงานจริง,
  PII, secret หรือ proprietary artifact และต้องหยุด runtime หลังจบ demo
- Dataset/image อาจมีข้อมูลทรัพย์สินทางปัญญา ต้องมี access control, encryption, retention และ audit policy

## Security release gate

ก่อน production ต้องทำ dependency/SBOM scan, container scan, signed provenance, penetration test,
incident/rollback runbook, privacy review และทดสอบ resource-abuse cases บน deployment target

## Implementation references

- [NumPy `load` security guidance](https://numpy.org/doc/stable/reference/generated/numpy.load.html)
  ระบุความเสี่ยงของ pickle และรองรับ `allow_pickle=False`
- [Pillow image security behavior](https://pillow.readthedocs.io/en/stable/reference/Image.html)
  อธิบาย decompression-bomb warning/error และ pixel-limit protection
