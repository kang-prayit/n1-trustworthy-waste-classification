# Preprocessing summary (computed from the released artifacts)

| Quantity | realwaste | trashnet | trashbox |
|---|---:|---:|---:|
| Classes | 9 | 6 | 7 |
| Images (after quarantine) | 4752 | 2527 | 17853 |
| Quarantined corrupt files | 0 | 0 | 3 |
| Duplicate groups (size>=2) | 16 | 39 | 855 |
| Images in duplicate groups | 63 | 79 | 1889 |
| % images in duplicate groups | 1.33 | 3.13 | 10.58 |
| Largest group | 16 | 3 | 53 |
| Redundant images (N - #groups) | 47 | 40 | 1034 |
| Images after dedup (1 per group) | 4705 | 2487 | 16819 |
| Exact MD5 duplicate images | 0 | 6 | 724 |
| Exact MD5 groups | 0 | 3 | 344 |
| pHash&dHash pairs | 2 | 29 | 842 |
| DINOv2 pairs (cos>=0.95, same class) | 92 | 31 | 1360 |
| Cross-class duplicate pairs | 0 | 3 | 11 |
|   of which exact MD5 | 0 | 3 | 3 |
| Mixed-label groups | 0 | 3 | 7 |
| Naive 70/15/15: % test items with dup group in train | 1.26 | 1.84 | 8.55 |
| Naive + offline aug x3: % test items with source image in train | 97.41 | n/a | n/a |
| TrashBox official test with dup in train (% , all criteria) | n/a | n/a | 8.37 |
| TrashBox official val with dup in train (% , all criteria) | n/a | n/a | 9.1 |
| TrashBox official test with dup in train (%, MD5 or pHash&dHash only) | n/a | n/a | 6.36 |
| TrashBox official val with dup in train (%, MD5 or pHash&dHash only) | n/a | n/a | 7.13 |
| TrashBox official test with exact copy in train (%) | n/a | n/a | 2.9 |
| TrashBox official val with exact copy in train (%) | n/a | n/a | 3.54 |
