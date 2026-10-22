Speed on **rishab-Vostro-3590** (x86_64, 8 threads), 320x240

| stage | ms per frame |
|---|---|
| stereo depth (SGBM) | 17.2 |
| optical flow (DIS) | 2.6 |
| detection (yolo11n.pt) | 26.1 |
| prediction, 5 people (social GRU) | 1.8 |
| planning (91 candidate paths) | 0.8 |
| **total** | **48.6** (20.6 fps) |
