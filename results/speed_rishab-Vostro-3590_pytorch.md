Speed on **rishab-Vostro-3590** (x86_64, 8 threads), 320x240

| stage | ms per frame |
|---|---|
| stereo depth (SGBM) | 12.1 |
| optical flow (DIS) | 1.9 |
| detection (yolo11n.pt) | 26.5 |
| prediction, 5 people (social GRU) | 2.3 |
| planning (91 candidate paths) | 1.0 |
| **total** | **43.7** (22.9 fps) |
