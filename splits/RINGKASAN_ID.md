# Ringkasan (Bahasa Indonesia)

Paket ini berisi seluruh artefak pra-pemrosesan untuk makalah *Leakage-Aware Re-evaluation, Confidence
Calibration, and Conformal Selective Prediction for Trustworthy Real-World Waste Classification*. Paket ini
**tidak memuat gambar**, hanya metadata: hash, grup duplikat, fold, dan statistik. Isinya:

- **Karantina file rusak.** Ada 3 gambar train TrashBox yang tidak bisa didekode PIL (cardboard 1075,
  e-waste 1719, dan paper 2273 yang berukuran 0 byte). Ketiganya dikeluarkan sebelum indexing.
- **Deteksi duplikat.** Kriterianya MD5 (salinan persis), pHash≤6 dan dHash≤6, serta cosine DINOv2 ViT-S/14
  ≥0,95 (hanya dalam kelas yang sama). Grup duplikat dibentuk sebagai komponen terhubung (union–find).
- **Hasil.** Persentase gambar yang berada dalam grup duplikat:
  - RealWaste: 63 gambar (1,33 %, 16 grup)
  - TrashNet: 79 gambar (3,13 %, 39 grup)
  - TrashBox: 1.889 gambar (10,58 %, 855 grup; 724 di antaranya salinan persis)
- **Label ganda.** TrashNet memiliki 3 salinan persis yang dilabeli dua kelas berbeda. TrashBox memiliki 3
  salinan persis lintas kelas, ditambah 8 pasangan hash lintas kelas yang setelah dicek visual kebanyakan
  *false positive*: objek berbeda dengan latar putih.
- **Kebocoran split resmi TrashBox.** Sebanyak 8,37 % gambar test resmi dan 9,10 % gambar val resmi punya
  duplikat di train. Untuk salinan persis saja, angkanya 2,90 % dan 3,54 %.
- **Split naive 70/15/15.** Proporsi item test yang grupnya juga ada di train: 1,26 % (RealWaste), 1,84 %
  (TrashNet), dan 8,55 % (TrashBox). Pada protokol "augmentasi dulu baru split" (RealWaste aug3), angkanya
  97,41 %.
- **Split grouped 5-fold** (StratifiedGroupKFold, seed 2026). Peran tiap gambar adalah train / cal_ts /
  cal_conf / test, dan tidak ada grup yang muncul di dua peran sekaligus.

File utama ada di `preprocessing_summary.md` dan `<dataset>/*.csv`. Contact sheet (thumbnail gambar) tidak disertakan karena lisensi gambar; buat ulang dengan `scripts/make_contact_sheets.py`. Cara reproduksi
dijelaskan di `README_preprocessing.md` bagian 4.
